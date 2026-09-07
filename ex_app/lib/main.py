#
# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
import asyncio
import contextlib
import logging
import os
import threading
import traceback
from contextlib import asynccontextmanager
from threading import Event
from time import perf_counter, sleep

import torch
from PIL import Image
from fastapi import FastAPI
from nc_py_api import NextcloudApp
from pdf2image import convert_from_path, pdfinfo_from_path
from nc_py_api.ex_app import AppAPIAuthMiddleware, LogLvl, get_computation_device, run_app, set_handlers
from nc_py_api.ex_app.providers.task_processing import TaskProcessingProvider
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from ex_app.lib.ocs import get_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def log(nc, level, content):
    logger.log((level+1)*10, content)
    if level < LogLvl.WARNING:
        return
    try:
        asyncio.run(nc.log(level, content))
    except:
        pass

TASKPROCESSING_PROVIDER_ID = 'ocr_paddle:ocr'

MODEL_NAME = "PaddlePaddle/PaddleOCR-VL"
# Pin the model revision so the remote code (trust_remote_code) cannot drift.
# Newer revisions switched to transformers' create_causal_mask(inputs_embeds=...)
# helper, which is incompatible with the pinned transformers version and breaks
# inference. This revision uses the self-contained _update_causal_mask path.
MODEL_REVISION = "be8ed7492f996cb9e0148aa0c97567f2f7bddfc5"

# Resolution used to rasterize PDF pages. 200dpi is the usual sweet spot for
# OCR: high enough to keep small print legible, low enough that a page image
# stays a manageable size.
PDF_DPI = int(os.environ.get("OCR_PDF_DPI", "200"))
# Safety net for very long documents: tasks are processed serially by a single
# worker, so one huge PDF would otherwise block every other queued task for
# hours. Pages past the limit are skipped and the truncation is reported both
# in the log and in the returned text. Set to 0 to read every page.
MAX_PDF_PAGES = int(os.environ.get("OCR_MAX_PDF_PAGES", "50"))
# Upper bound on the long edge of a rasterized page. Rendering a large-format
# page (a poster or a plan) at PDF_DPI can produce a huge bitmap, so scale
# those down before OCR. Generous enough that ordinary A4/A3 pages at 200dpi
# are never touched, and the model's processor resizes its input anyway.
MAX_PAGE_PIXELS = int(os.environ.get("OCR_MAX_PAGE_PIXELS", "4000"))

def load_model():
    if get_computation_device().lower() == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = model.to('cuda').eval()
        device = 'cuda'
    else:
        # Cpu does not support fp16
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            trust_remote_code=True,
        )
        model = model.to("cpu").eval()
        device = 'cpu'

    processor = AutoProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True)
    return model, processor, device


app_enabled = Event()
TRIGGER = Event()

WAIT_INTERVAL = 5
WAIT_INTERVAL_WITH_TRIGGER = 5 * 60

@asynccontextmanager
async def lifespan(app: FastAPI):
    set_handlers(
        app,
        enabled_handler,
        trigger_handler=trigger_handler,
    )
    nc = NextcloudApp()
    if nc.enabled_state:
        app_enabled.set()
    start_bg_task()
    yield


APP = FastAPI(lifespan=lifespan)
APP.add_middleware(AppAPIAuthMiddleware)  # set global AppAPI authentication middleware

def start_bg_task():
    t = threading.Thread(target=background_thread_task)
    t.start()

def background_thread_task():
    nc = NextcloudApp()
    while not app_enabled.is_set():
        sleep(5)

    model, processor, device = load_model()

    while True:
        if not app_enabled.is_set() or model is None or processor is None:
            sleep(30)
            continue
        try:
            next = nc.providers.task_processing.next_task([TASKPROCESSING_PROVIDER_ID], ['core:image2text:ocr'])
            if not 'task' in next or next is None:
                wait_for_task()
                continue
            task = next.get('task')
        except Exception as e:
            print(str(e))
            log(nc, LogLvl.ERROR, str(e))
            wait_for_task(30)
            continue
        try:
            log(nc, LogLvl.INFO, f"Next task: {task['id']}")

            log(nc, LogLvl.INFO, "Running OCR")

            time_start = perf_counter()
            fileIds = task.get("input").get('input')
            outputs = []
            for index, fileId in enumerate(fileIds):
                # Give each input file an equal slice of the progress bar; a
                # multi-page PDF then advances within its own slice per page.
                outputs.append(process_file(
                    device, fileId, model, nc, processor, task,
                    progress_start=100.0 * index / len(fileIds),
                    progress_end=100.0 * (index + 1) / len(fileIds),
                ))
            log(nc, LogLvl.INFO, f"OCR finished: {perf_counter() - time_start}s")


            nc.providers.task_processing.report_result(
                task["id"],
                { 'output': outputs },
            )

        except Exception as e:  # noqa
            print(str(e) + "\n" + "".join(traceback.format_exception(e)))
            try:
                log(nc, LogLvl.ERROR, str(e))
                nc.providers.task_processing.report_result(task["id"], None, str(e))
            except:
                pass
            wait_for_task(30)


def is_pdf(file_name):
    # The file is downloaded to a temporary file without an extension, and the
    # task input carries no mime type, so sniff the content instead. The header
    # is normally at offset 0, but the spec tolerates leading bytes and so does
    # poppler, so scan the start of the file rather than matching a prefix.
    with open(file_name, "rb") as f:
        return b"%PDF-" in f.read(1024)


def load_pages(file_name):
    """Return (total_pages, pages_to_read, page_image_iterator) for a task input file.

    Plain images are a single page. PDFs are rasterized one page at a time so
    that memory use stays flat regardless of document length, instead of
    holding every rendered page at once.
    """
    if not is_pdf(file_name):
        return 1, 1, iter((Image.open(file_name).convert("RGB"),))

    total_pages = pdfinfo_from_path(file_name)["Pages"]
    pages_to_read = min(total_pages, MAX_PDF_PAGES) if MAX_PDF_PAGES > 0 else total_pages

    def render_pages():
        for page in range(1, pages_to_read + 1):
            image = convert_from_path(
                file_name,
                dpi=PDF_DPI,
                first_page=page,
                last_page=page,
            )[0].convert("RGB")
            # thumbnail() preserves the aspect ratio and never upscales, so
            # normally sized pages pass through unchanged.
            image.thumbnail((MAX_PAGE_PIXELS, MAX_PAGE_PIXELS), Image.LANCZOS)
            yield image

    return total_pages, pages_to_read, render_pages()


def run_ocr(device, model, processor, image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "OCR:"}
            ]
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    outputs = model.generate(**inputs, max_new_tokens=1024)
    outputs = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    # The decoded text echoes the prompt, so keep only the assistant's turn.
    # Fall back to the full decode rather than failing the whole document if a
    # single page comes back without the marker.
    marker = 'Assistant: '
    return outputs.split(marker, 1)[1] if marker in outputs else outputs


def process_file(device, fileId, model, nc, processor, task, progress_start, progress_end):
    file_name = get_file(nc, task["id"], fileId)
    try:
        total_pages, pages_to_read, pages = load_pages(file_name)
        if pages_to_read < total_pages:
            log(nc, LogLvl.WARNING,
                f"File {fileId} has {total_pages} pages, reading only the first {pages_to_read} "
                f"(OCR_MAX_PDF_PAGES={MAX_PDF_PAGES})")

        texts = []
        for page_number, page in enumerate(pages, start=1):
            texts.append(run_ocr(device, model, processor, page))
            nc.providers.task_processing.set_progress(
                task['id'],
                progress_start + (progress_end - progress_start) * page_number / pages_to_read,
            )

        if pages_to_read < total_pages:
            texts.append(f"[Only the first {pages_to_read} of {total_pages} pages were processed.]")
        return "\n\n".join(texts)
    finally:
        # get_file() streams into a NamedTemporaryFile(delete=False), so nothing
        # else removes it. PDFs make this leak matter more than single images.
        with contextlib.suppress(OSError):
            os.unlink(file_name)


async def enabled_handler(enabled: bool, nc: NextcloudApp) -> str:
    # This will be called each time application is `enabled` or `disabled`
    # NOTE: `user` is unavailable on this step, so all NC API calls that require it will fail as unauthorized.
    print(f"enabled={enabled}")
    if enabled:
        log(nc, LogLvl.WARNING, f"Enabled: {nc.app_cfg.app_name}")
        await nc.providers.task_processing.register(TaskProcessingProvider(
            id=TASKPROCESSING_PROVIDER_ID,
            name='Nextcloud Local OCR: Paddle OCR',
            task_type='core:image2text:ocr',
            # Only an ETA hint shown in the UI, nothing enforces it. Raised
            # because a multi-page PDF takes considerably longer than one image.
            expected_runtime=300,
        ))
        app_enabled.set()
    else:
        await nc.providers.task_processing.unregister(TASKPROCESSING_PROVIDER_ID, True)
        log(nc, LogLvl.WARNING, f"Disabled {nc.app_cfg.app_name}")
        app_enabled.clear()
    # In case of an error, a non-empty short string should Yesbe returned, which will be shown to the NC administrator.
    return ""


def trigger_handler(providerId: str):
    # This will only get called on Nextcloud 33+
    TRIGGER.set()

# Waits for `interval` seconds or `WAIT_INTERVAL` seconds
# if `interval` is not set. If TRIGGER gets set in the meantime,
# WAIT_INTERVAL gets overriden with WAIT_INTERVAL_WITH_TRIGGER which should be longer
def wait_for_task(interval = None):
    global TRIGGER
    global WAIT_INTERVAL
    global WAIT_INTERVAL_WITH_TRIGGER
    if interval is None:
        interval = WAIT_INTERVAL
    if TRIGGER.wait(timeout=interval):
        WAIT_INTERVAL = WAIT_INTERVAL_WITH_TRIGGER
    TRIGGER.clear()

if __name__ == "__main__":
    # Wrapper around `uvicorn.run`.
    # You are free to call it directly, with just using the `APP_HOST` and `APP_PORT` variables from the environment.
    run_app("main:APP", log_level="trace")
