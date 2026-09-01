# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from openai import OpenAI
from transformers.utils.versions import require_version


require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")


def main():
    client = OpenAI(
        api_key="{}".format(os.getenv("API_KEY", "0")),
        base_url="http://localhost:{}/v1".format(os.getenv("API_PORT", 8000)),
    )
    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Output the color and number of each box."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/boxes.png"},
                },
            ],
        }
    )
    result = client.chat.completions.create(messages=messages, model="test")
    if not result.choices or result.choices[0].message is None:
        raise SystemExit("Round 1: no response received")
    messages.append(result.choices[0].message)
    print("Round 1:", result.choices[0].message.content)
    # The image shows a pyramid of colored blocks with numbers on them. Here are the colors and numbers of ...
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What kind of flower is this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/flowers.jpg"},
                },
            ],
        }
    )
    result = client.chat.completions.create(messages=messages, model="test")
    if not result.choices or result.choices[0].message is None:
        raise SystemExit("Round 2: no response received")
    messages.append(result.choices[0].message)
    print("Round 2:", result.choices[0].message.content)
    # The image shows a cluster of forget-me-not flowers. Forget-me-nots are small ...


if __name__ == "__main__":
    main()
