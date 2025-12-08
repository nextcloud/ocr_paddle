import asyncio
import logging
import threading
import traceback
from contextlib import asynccontextmanager
from threading import Event
from time import perf_counter, sleep

import torch
from PIL import Image
from fastapi import FastAPI
from nc_py_api import NextcloudApp
from nc_py_api.ex_app import AppAPIAuthMiddleware, LogLvl, get_computation_device, run_app, set_handlers
from nc_py_api.ex_app.providers.task_processing import ShapeDescriptor, ShapeType, TaskProcessingProvider, TaskType
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

def load_model():
    if get_computation_device().lower() == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(
            "PaddlePaddle/PaddleOCR-VL",
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model = model.to('cuda').eval()
        device = 'cuda'
    else:
        # Cpu does not support fp16
        model = AutoModel.from_pretrained(
            "PaddlePaddle/PaddleOCR-VL",
            trust_remote_code=True
        )
        model = model.to("cpu").eval()
        device = 'cpu'

    processor = AutoProcessor.from_pretrained("PaddlePaddle/PaddleOCR-VL", trust_remote_code=True)
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
            fileId = task.get("input").get('input')
            file_name = get_file(nc, task["id"], fileId)
            image = Image.open(file_name).convert("RGB")
            nc.providers.task_processing.set_progress(task['id'], 15)

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
            nc.providers.task_processing.set_progress(task['id'], 30)
            outputs = model.generate(**inputs, max_new_tokens=1024)
            nc.providers.task_processing.set_progress(task['id'], 75)
            outputs = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            nc.providers.task_processing.set_progress(task['id'], 95)
            log(nc, LogLvl.INFO, f"OCR finished: {perf_counter() - time_start}s")


            nc.providers.task_processing.report_result(
                task["id"],
                {'output': outputs.split('Assistant: ', 1)[1]},
            )

        except Exception as e:  # noqa
            print(str(e) + "\n" + "".join(traceback.format_exception(e)))
            try:
                log(nc, LogLvl.ERROR, str(e))
                nc.providers.task_processing.report_result(task["id"], None, str(e))
            except:
                pass
            wait_for_task(30)



async def enabled_handler(enabled: bool, nc: NextcloudApp) -> str:
    # This will be called each time application is `enabled` or `disabled`
    # NOTE: `user` is unavailable on this step, so all NC API calls that require it will fail as unauthorized.
    print(f"enabled={enabled}")
    if enabled:
        await nc.log(LogLvl.WARNING, f"Enabled: {nc.app_cfg.app_name}")
        await nc.providers.task_processing.register(TaskProcessingProvider(
            id=TASKPROCESSING_PROVIDER_ID,
            name='Nextcloud Local OCR: DeepSeek OCR',
            task_type='core:image2text:ocr',
            expected_runtime=120,
        ))
        app_enabled.set()
    else:
        await nc.providers.task_processing.unregister(TASKPROCESSING_PROVIDER_ID, True)
        nc.log(LogLvl.WARNING, f"Disabled {nc.app_cfg.app_name}")
        app_enabled.clear()
    # In case of an error, a non-empty short string should be returned, which will be shown to the NC administrator.
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
