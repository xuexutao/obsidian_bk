## 一、抖音3D资产VLM

### 1.1 豆包微调模型

微调豆包模型主要是通过方舟平台，方舟平台主界面，具身使用 EmbodiedAI项目：

https://cloud.bytedance.net/ark/region:ark+cn-beijing/overview?arkAccountId=2100583685&arkProjectName=ark_project_20251111_3cgql&type=&x-bc-region-id=bytedance&x-resource-account=public

如果要微调豆包，只需要准备好训练用的 jsonl，就可以在这个平台上进行微调了

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=Mjc3ZDFlMjdkMzFmN2Q1M2NkZTcwNmE4NWIwODI5NDZfc2Q3SmlSd3l3NGZLNFVDVjJocUg5MjBzTlJ0ZUd0TFZfVG9rZW46SVFBTGJCRllOb2ZjVWZ4REc5emNyaDhDbjFHXzE3ODI5ODIyNTg6MTc4Mjk4NTg1OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

可以参考我接触的 图文微调 VLM， 步骤如下：

1、准备jsonl，jsonl中每一行的json结构如下：

```JSON
{
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "##角色##\n你是一个多视图几何领域的专家，接下来我给你输入一个物体的多视角图片，请你基于这些多视角图片对该3D资产.........prompts..........断\": [\n\"包含贴图\",\n\"存在背景板、底版\",\n\"属于建筑房屋类\"\n],\n\"基础色贴图光影效果\": 1\n}\n"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "tos://ark-auto-2100583685-cn-beijing-ark-project-20251111-3cgql/xxt/douYin6WTrain/geo_render_image_train/0a679ef337a04dee98ec55a49bb9d1c7.png"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "tos://ark-auto-2100583685-cn-beijing-ark-project-20251111-3cgql/xxt/douYin6WTrain/tex_albedo_image_train/0a679ef337a04dee98ec55a49bb9d1c7.png"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "tos://ark-auto-2100583685-cn-beijing-ark-project-20251111-3cgql/xxt/douYin6WTrain/tex_render_image_train/0a679ef337a04dee98ec55a49bb9d1c7.png"
                    }
                }
            ]
        },
        {
            "role": "assistant",
            "content": "{\n\"物体复杂度\":3,\n\"贴图精细度\":3,\n\"结构合理性\":4,\n\"表面光滑度\":2,\n\"贴图颜色丰富度\":4,\n\"几何整体质量\":3,\n\"贴图光影质感\":4,\n\"贴图整体质量\":3,\n\"特殊情况判断\":[\"包含贴图\",\"属于低面数模型(Low Poly)\",\"存在摆放错误\"],\n\"基础色贴图光影效果\":3\n}\n"
        }
    ],
    "thinking": "disabled"
}
```

2、把jsonl上传到 方舟TOS桶 中，代码如下，这个代码版本比较早了，模型微调过程中，需要对图片有一个可访问的鉴权地址（http）或者tos开头的地址，两者都可以，具体的如何配置环境等可以看：[方舟平台模型训练与推理](https://bytedance.larkoffice.com/wiki/BXXDwNOJoibXIKkavFMcTlKlnGe?from=from_copylink)

```Python
import os
import ark_tos
import concurrent.futures
import threading
import itertools

AK = "service_account=xxt_test;main_account_id=2100583685;"
SK = "559b67a861b93ba7199fc62f169f2d00"
BUCKET_NAME = 'ark-auto-2100583685-cn-beijing-ark-project-20251111-3cgql'

thread_local = threading.local()

def get_tos_client(bucket_name):
    if not hasattr(thread_local, "client"):
        ak = AK
        sk = SK
        thread_local.client = ark_tos.new_tos_cn_client(ak=ak, sk=sk, bucket=bucket_name)
    return thread_local.client

def generate_upload_tasks(filePath, object_key_prefix):
    for dir_name in os.listdir(filePath):
        path = os.path.join(filePath, dir_name)
        if os.path.isdir(path):
            for file_name in os.listdir(path):
                object_key = f"{object_key_prefix}/{dir_name}/{file_name}"
                local_file_path = os.path.join(path, file_name)
                yield object_key, local_file_path

def upload_worker(task, bucket_name):
    key, file_path = task
    try:
        client = get_tos_client(bucket_name)
        client.put_object_from_file(key=key, local_file_path=file_path)
        return (file_path, True, None)
    except Exception as e:
        return (file_path, False, e)

def upload_tos(bucket_name, object_key_prefix, filePath, max_workers=10):
    tasks_generator = generate_upload_tasks(filePath, object_key_prefix)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(upload_worker, tasks_generator, itertools.repeat(bucket_name))

        for file_path, success, error in results:
            if not success:
                print(f"Failed to upload {file_path}: {error}")

def generation_signed_image_path(name):
    client = ark_tos.new_tos_cn_client(ak=AK, sk=SK, bucket=BUCKET_NAME)
    signed_url1 = client.pre_signed_url(key=f"xxt/douyinvlm_test2/geo_render_image_test/{name}")
    signed_url2 = client.pre_signed_url(key=f"xxt/douyinvlm_test2/tex_albedo_image_test/{name}")
    signed_url3 = client.pre_signed_url(key=f"xxt/douyinvlm_test2/tex_render_image_test/{name}")
    return signed_url1, signed_url2, signed_url3

                
if __name__ == "__main__":
    # ----------------------------------------------------
    # upload batch file to bucket
    # ----------------------------------------------------
    # OBJECT_KEY_PREFIX = 'xxt/douyinvlm_test2'
    # FILEPATH = "test2"
    # upload_tos(
    #     bucket_name=BUCKET_NAME, 
    #     object_key_prefix=OBJECT_KEY_PREFIX, 
    #     filePath=FILEPATH,
    #     max_workers=50
    # )
    # print("Done!")

    # ----------------------------------------------------
    # upload single JSONL file
    # ----------------------------------------------------
    client = ark_tos.new_tos_cn_client(ak=AK, sk=SK, bucket=BUCKET_NAME)
    client.put_object_from_file(key="xxt/douyinvlm_test2/batch_reasoning_test2.jsonl", local_file_path="./batch_reasoning_test2.jsonl")
    # print(generation_signed_image_path("00009d1ad5372fb9871a6b2d3f68669e.png"))

    # ----------------------------------------------------
    # accept signed file_url
    # ----------------------------------------------------
    # client = ark_tos.new_tos_cn_client(ak=AK, sk=SK, bucket=BUCKET_NAME)
    # signed_url1 = client.pre_signed_url(key="xxt/douyinvlm/geo_render_images/0a0bc2921e5246a28732bf5584c251d1.png")
    # signed_url2 = client.pre_signed_url(key="xxt/douyinvlm/tex_albedo_images/0a0bc2921e5246a28732bf5584c251d1.png")
    # signed_url3 = client.pre_signed_url(key="xxt/douyinvlm/tex_render_images/0a0bc2921e5246a28732bf5584c251d1.png")
    # print(signed_url1)
    # print(signed_url2)
    # print(signed_url3)
```

3、准备精调

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=NjMyYzQzM2YzNzI4ZjlmZjNlNjc2MmQwMGUwYzM5MjZfR3F5UTl0VURvckhrbjVyOUcwSzJoUVlkYnh3N1FDV2tfVG9rZW46QnFIMGJGVDdPb2N3d1J4Q2hzUWNYWWxmblVmXzE3ODI5ODIyNTg6MTc4Mjk4NTg1OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=ZGFlMDcyY2U0YTY5YThlNjQ2ODNmNjg1NDRlYTlhMGFfNFVacm4xT21qZWh0blpJNkR0WExINjNnZ3pJUGJSU0FfVG9rZW46SE1HcWI3VFBVb1pON2p4SExQRmNyY0pmblBoXzE3ODI5ODIyNTg6MTc4Mjk4NTg1OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=ZjllMGVjNDJjODA5ZmY3ZjY5NDYyZDIxZDUzMTM2ZGNfVTNNRGR0cm5Mc2NYZ2VEUEIzNkVXa0VMZVRIbmxPZVVfVG9rZW46RFNBdWJiVUVxb3hOWUh4VWJ0V2NaOUJ3bnpoXzE3ODI5ODIyNTg6MTc4Mjk4NTg1OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=OTNmMzZjYjNmOTQyZGY0NDg0NWU2Zjg5MTc1YmJjYmZfYVV6eWp3SmV5V2htMWFibTE0ODVuZmM4OUdTZWJYZ2pfVG9rZW46RndtTWJZRGQzb0xQTHB4WXJIZmM5cXhibmVoXzE3ODI5ODIyNTg6MTc4Mjk4NTg1OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

### 1.2 豆包离线推理

离线推理有两种，BatchChat和BatchJob，其中

- BatchChat就是直接用一个 离线的ak/sk，使用持续的高并发、长超时、多重试的策略在资源池中竞争资源
- BatchJob是通过方舟平台内部的任务优先级，让他们内部人员自动给我们分配资源和自动处理并发方式，只要提交了，且有优先级的话，一晚上可交付的数据量很大，主要推荐BatchJob方式，需要先填写**资源需求收集表**，然后就是和1.1 相似的步骤，把数据上传到 方舟tos 桶中，且构造一下 jsonl
    - 提优表单：[方舟批量推理资源需求收集表](https://bytedance.larkoffice.com/wiki/NApgw0gDUilzRSk1gltc2tbhnnf?from=from_copylink)，填完后还需要单独发送给对应负责人，才能提优

方舟tos上传、鉴权的说明文档在：[字节云方舟批量推理API操作指南](https://bytedance.larkoffice.com/wiki/Q0SMwVvcEiHlPNkS2fjcjw0unih#share-CLvpdeiisoaLYKxFbm5cW8LcnGd)

【方舟平台的tos，是一个文件存储bucket，方舟中的模型只能看到这里的数据，所以需要把本地文件或者服务器文件上传到 tos 桶中，并且通过鉴权得到唯一的 http 链接地址或者tos开头的地址，让 VLM 模型拿到数据。】

示例代码：

https://code.byted.org/aidp-playground/algo-embodiedintelligence-AssertsProcess/blob/dev_xutao/upload_tos_ray/upload_ray.py?ref_type=heads

这个代码主要功能是视频理解，同时做一下三步：

1、上传 mp4 视频到 tos 桶，2、鉴权得到 mp4 访问链接，3、生成 batch job 使用的 jsonl

```JSON
{
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": prompt,
        },
        {
            "type": "video_url",
            "video_url": {
                "url": item["http_url"],
            },
        },
    ],
}
```

平台操作：

（支持使用平台模型、也支持微调后的模型【需要在相应界面量化后才可使用】）

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=ODZjZmI2MDVkYjg2ZmM3ODcyYjFiMTEwYjU5MzZkOThfUURuTXNvY0FldnBrZG83RWZ1ZG5NdE5KUlMxYnpGbFFfVG9rZW46VUdEcmJyNlhCb3Via0J4bnpsT2NLazh0bjNjXzE3ODI5ODIyNTg6MTc4Mjk4NTg1OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=ODRkMTIxYzdjYTdlMTk1YWUwMDMwZDQyZDE5ZGViMTVfcEUzYnNsdUtxMVlmYjBFanY0b2h4NFFvTGNLZFBWbVhfVG9rZW46TlowMGJORndQb2ZKaUZ4OGNKNGNMM2h5bnlnXzE3ODI5ODIyNTg6MTc4Mjk4NTg1OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=ZmUyYWM3YzRkMGVkMDMzMGY1MWM5NjI4NTZkOTQ0ZjRfaTZhblFZc3doNW5STmFKbW9aY3pSajZnbGtpYWdzdjFfVG9rZW46TXRkUWJwRmpUb2NrV3J4MWtPVWN2UlFwbmtoXzE3ODI5ODIyNTg6MTc4Mjk4NTg1OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

## 二、TableVerse

### 2.1 资产渲染

TableVerse 渲染图 存放位置在：hdfs://haruna/home/byte_isp_data/user/wangboyuan/real2sim_two_phases/_stage1/{scene_id}/renders/{object_name}/，示例如下：

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=ODM2MDhkOGJiOTY4MjUyYWVhMjAyNzgxYmZhNGRlNjNfMHVqMGU1ZDRzakx4YzFPRW1PZFVTUzU4RHFWMUl2SzdfVG9rZW46Qkx5bmJmelcwb3N5djN4YThSWGNwZEJMbnRlXzE3ODI5ODIyNTg6MTc4Mjk4NTg1OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

对于每个资产进行多视图渲染，渲染结果如下：

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=ZWQ4MjQxYmMxNjJlMGZhNGJhOWJhNTYxY2ZlYjc4MDdfalRmazJPamRnamJTY29TdDZsT1ljMzFETTJEODNtZklfVG9rZW46V3l3V2J1UFc2b3RYYmp4bHlVZmNnYlV4bktOXzE3ODI5ODIyNTg6MTc4Mjk4NTg1OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=MDBhZDMzYjhmODAyMzhmYTAyZDQ0NjhmNzdlMmIwNjRfTHlMY3p1Qml4WmNONjBIcHk1N1c4U2tVdlRnT3dsTEVfVG9rZW46UEFtWGI0RGsyb2pHQ1h4TVA1ZmNxNE50bmhjXzE3ODI5ODIyNTg6MTc4Mjk4NTg1OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=OTRjODUwMWEyZDk4MDA2NWM2YTY0MTUwY2E3YmIxYWNfa0o0SlBFaVBRVkVKOWpNTG5WNlFqT0l4NlU1cElnR0tfVG9rZW46SEI1S2JjNFFqbzl2Um54bjFpR2NvVVZ2bnNiXzE3ODI5ODIyNTg6MTc4Mjk4NTg1OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

除此之外还在MIPP创建了一个渲染算子 aidp/render_single_assert：

https://cloud.bytedance.net/mipp/operators/operator_details?_clu=default&_idc=lf&domainId=aidp&name=aidp%2Frender_single_assert&type=Type_BATCH&x-bc-vdc=http&x-bc-vregion=China-North&x-resource-account=public&x-bc-region-id=bytedance

示例输入：hdfs://haruna/home/byte_isp_data/user/xuexutao/bench_dataset/MIPP_PROCESS_LOG/hy3dbench_ply.parquet

https://cloud.bytedance.net/mipp/batch_jobs/detail?_clu=default&_clu=default&_idc=lf&_idc=lf&domainId=aidp&id=20260511%3A7638348687261094182&x-bc-vdc=http&x-bc-vregion=China-North&x-resource-account=public&x-bc-region-id=bytedance

### 2.2 资产打标

主要是针对渲染得到的三张图片进行 VLM 资产属性打标，打标的属性可以参考：

```JSON
{{
  "classification": {{
    "PrimaryCategory": "",
    "SecondaryCategory": "",
    "AssetLabel": "",
    "Style": [],
    "Complexity": "",
    "Classification_reasoning": ""
  }},
  "attributes": {{
    "can_place_on_desk": false,
    "can_be_grasped": false,
    "is_character_asset": false,
    "is_composite_asset": {{
      "visible_items": ["list", "of", "identifiable", "items", "in", "the", "image"],
      "reasoning": "Explain step-by-step whether the items form one cohesive entity or are logically distinct independent objects based on the guidelines.",
      "classification": "Single Object" | "Multiple Objects"
    }},
    "attributes_reasoning": ""
  }},
  "physics_output": {{
    "absolute_scale_cm": {{"L":0,"W":0,"H":0}},
    "relative_scale_ratio_lwh": [0,0,0],
    "mass_kg": 0.0,
    "asset_scope": "object|scene|unknown",
    "asset_description": "",
    "articulation": {{
      "has_articulation": false,
      "joint_types": [],
      "count_estimate": 0
    }},
    "contact_properties": {{
      "static_friction": 0.0,
      "dynamic_friction": 0.0
    }}
  }},
  "overall_confidence": 0.0,
  "reasoning": ""
}}
```

代码如下：

https://code.byted.org/aidp-playground/algo-embodiedintelligence-3DProcessPlayground/tree/xutao_dev/TableVerse/retrieve?ref_type=heads

### 2.3 检索库构建

**构建**

利用 Doubao embedding 模型对上一步打标得到的json和渲染得到的图片进行向量提取，且存储在本地vectors.f32中，后续可以通过脚本从中检索资产，检索结果就是配套的meta.json中的信息。

检索库构建代码：

https://code.byted.org/aidp-playground/algo-embodiedintelligence-AssertsProcess/blob/dev_xutao/RetrieveHome/build_index.py?ref_type=heads

**检索**

检索代码位置：hdfs://haruna/home/byte_isp_data/user/xuexutao/AssertRetrieve/index，两种检索方式：

````Markdown
## 一、直接查找

### 1、文本查询

```bash
python find_by_text_image.py --text "apple" --topk 10

# topk： 默认前20
```

结果：

```json
{
    "rank": 1,
    "local_rank": 1,
    "score": 0.5792946815490723,
    "query_mode": "text",
    "prompt": "apple",
    "index_name": "trellis_glb",
    "sha256": "894bf7770ed9c1f1b68506df32b1d567eddee9d9527b51324a87c6b098d0c884",
    "asset_path": "hdfs://haruna/home/byte_isp_data/xuexutao/bench_dataset/Trellis-500k/ObjaverseXL_sketchfab/raw/glbs/000-103/e13413e3872a4d0a90758c5f4b9076b3.glb",
    "albedo_path": "hdfs://haruna/home/byte_isp_data/xuexutao/bench_dataset/bench_render/trellis/894bf7770ed9c1f1b68506df32b1d567eddee9d9527b51324a87c6b098d0c884/894bf7770ed9c1f1b68506df32b1d567eddee9d9527b51324a87c6b098d0c884_albedo.jpg",
    "phy_attribute_path": "hdfs://haruna/home/byte_isp_data/xuexutao/bench_dataset/bench_render/trellis/894bf7770ed9c1f1b68506df32b1d567eddee9d9527b51324a87c6b098d0c884/phy_attribute.json"
}
```

### 2、图片查询

```bash
python find_by_text_image.py --image /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-embodiedintelligence-3DProcessPlayground/AssertWareHouse/retrieve_script/apple.jpg
```

结果：

```json
{
    "rank": 1,
    "local_rank": 1,
    "score": 0.5471563339233398,
    "query_mode": "image",
    "prompt": null,
    "index_name": "trellis_glb",
    "sha256": "f8804d34fd0676926623c36bdc0680649ca3c7d71906c17e41c297c8dd9ea877",
    "asset_path": "hdfs://haruna/home/byte_isp_data/xuexutao/bench_dataset/Trellis-500k/ObjaverseXL_sketchfab/raw/glbs/000-139/ef8b3bdd350f4803af98be1b3fb2aae1.glb",
    "albedo_path": "hdfs://haruna/home/byte_isp_data/xuexutao/bench_dataset/bench_render/trellis/f8804d34fd0676926623c36bdc0680649ca3c7d71906c17e41c297c8dd9ea877/f8804d34fd0676926623c36bdc0680649ca3c7d71906c17e41c297c8dd9ea877_albedo.jpg",
    "phy_attribute_path": "hdfs://haruna/home/byte_isp_data/xuexutao/bench_dataset/bench_render/trellis/f8804d34fd0676926623c36bdc0680649ca3c7d71906c17e41c297c8dd9ea877/phy_attribute.json"
}
```

### 3、图文检索

```bash
python find_by_text_image.py --image /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-embodiedintelligence-3DProcessPlayground/AssertWareHouse/retrieve_script/apple.jpg --text "green apple"
```

结果：

```json
{
    "rank": 1,
    "local_rank": 1,
    "score": 0.5156591534614563,
    "query_mode": "mix",
    "prompt": "green apple",
    "index_name": "trellis_glb",
    "sha256": "2c809f6aa728957a5a334688f14770c057f1fffef02aa4aca5c2cbd4cb658030",
    "asset_path": "hdfs://haruna/home/byte_isp_data/xuexutao/bench_dataset/Trellis-500k/ObjaverseXL_sketchfab/raw/glbs/000-017/b9961fd91af445e18971c0d287df2c1a.glb",
    "albedo_path": "hdfs://haruna/home/byte_isp_data/xuexutao/bench_dataset/bench_render/trellis/2c809f6aa728957a5a334688f14770c057f1fffef02aa4aca5c2cbd4cb658030/2c809f6aa728957a5a334688f14770c057f1fffef02aa4aca5c2cbd4cb658030_albedo.jpg",
    "phy_attribute_path": "hdfs://haruna/home/byte_isp_data/xuexutao/bench_dataset/bench_render/trellis/2c809f6aa728957a5a334688f14770c057f1fffef02aa4aca5c2cbd4cb658030/phy_attribute.json"
}
```

## 二、服务查找

直接查找真正慢的是从磁盘读入检索库的过程，每次查找都需要读入检索库，而且hdfs的文件读取速度较慢。对比检索过程速度倒是很快，所以可以使用API服务的方式，首先运行server，变为常驻进程，后续调用时直接从内存中检索即可。

运行方式：

```Shell
# 安装依赖
pip install fastapi uvicorn

# 启动server
uvicorn server:app --host 0.0.0.0 --port 8000
```

如何调用：

```Shell
# 1、文本检索
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "modern office desk",
    "topk": 20
  }'

# 2、图片检索
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "image": "/mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-embodiedintelligence-3DProcessPlayground/AssertWareHouse/retrieve_script/apple.jpg",
    "topk": 20
  }'

# 3、图文检索
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "green apple",
    "image": "/mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-embodiedintelligence-3DProcessPlayground/AssertWareHouse/retrieve_script/apple.jpg",
    "topk": 20
  }'
```
````

目前可检索资产在：/mnt/hdfs/byte_isp_data/user/xuexutao/AssertRetrieve/index/mix2/ 里面，组成：

- TRELLIS: 22w6819
- partnet-mobility: 1716
- hybench: 5w0880

### 2.4 对比实验

主要对比三个模型：scenemaker、sam3d、MIDI。由于三个都是第三方库，暂时没有用 git 仓库，三个代码在：/mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-thirdparts 可以看到。

**安装**

前置环境 nvcc 12.1 (我一般只用这一个版本，torch 用 2.5.1)。环境安装脚本：

```Bash
cd ~
# 所有开发机使用同一个 .cache，可以免除很多模型下载
rm -rf ~/.cache/
sudo ln -s /mnt/bn/aidp-data-3d-lf1/xxt/.cache/ ~/.cache
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118 HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118 NO_PROXY=.byted.org

sudo apt-get install tmux -y
```

SceneMaker

```Bash
pip install uv                                                                                                                    
uv venv --python 3.10 scenemaker                                                                                                  
source scenemaker/bin/activate                                                                                                    
                                                                                                                                  
sudo apt update                                                                                                                   
sudo apt install -y libgl1-mesa-glx libglib2.0-0                                                                                  
                                                                                                                                  
cd /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-thirdparts/SceneMaker                                                       
sudo chown -R tiger:tiger /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-thirdparts/SceneMaker                                

# ===============================
# 安装环境
# ===============================
# 先装 torch 主栈                                                                                                                 
uv pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" --index-url https://download.pytorch.org/whl/cu121
                                                                                                                                  
uv pip install setuptools==65.5.0 wheel==0.38.4                                                                                   
                                                                                                                                  
# requirements 继续 no-deps                                                                                                       
uv pip install -r requirements.txt --no-deps                                                                                      
                                                                                                                                  
# xformers 用和 torch 2.5.1 匹配的版本                                                                                            
uv pip install "xformers==0.0.29.post1" --index-url https://download.pytorch.org/whl/cu121                                        
                                                                                                                                  
# 其他包                                                                                                                          
uv pip install warp-lang
uv pip install open-clip-torch --no-deps
uv pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html
uv pip install deepspeed
uv pip install pytorch_lightning
uv pip install "cubvh @ git+https://github.com/ashawkey/cubvh.git@8f0c777c2f9bd29b73931554d09269eed7023880" --no-build-isolation  
uv pip install "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" --no-build-isolation
uv pip install --no-cache-dir --reinstall "fastrlock>=0.5"
uv pip install --reinstall --no-deps "cupy-cuda12x==13.4.1"
                                                                                                                                  
cd Step1X-3D/step1x3d_texture/custom_rasterizer/
uv pip install -e . --no-build-isolation
cd ../differentiable_renderer/
uv pip install -e . --no-build-isolation
cd ../../../


# ===== 关键补丁：最后再把核心栈钉回去 =====                                                                                      
uv pip install --reinstall \
  "torch==2.5.1" \
  "torchvision==0.20.1" \
  "torchaudio==2.5.1" \
  --index-url https://download.pytorch.org/whl/cu121

uv pip install --reinstall "numpy==1.26.4"

uv pip install --reinstall \
  "xformers==0.0.29.post1" \
  --index-url https://download.pytorch.org/whl/cu121 
```

SAM3D

```Bash
# ==============================
# install 1
# ==============================
pip install uv
uv venv --python 3.10
source .venv/bin/activate
sudo apt update
sudo apt install -y libgl1-mesa-glx libglib2.0-0

uv pip install hatchling
uv pip install "pip<24"
uv pip install setuptools==65.5.0 wheel==0.38.4
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
cd /mnt/bn/isp-vladata-lq2/xuexutao/demo/sam-3d-objects
uv pip install -r requirements.txt --no-build-isolation

uv pip install hatch-requirements-txt editables
uv pip install -e '.[dev]' --no-build-isolation
uv pip install -e '.[p3d]' --no-build-isolation
uv pip install -e '.[inference]' --no-build-isolation
./patching/hydra

# kaolin
uv pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html
uv pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

# flash-atten
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


uv pip install "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7" --no-build-isolation
# uv pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git --no-build-isolation
uv pip install submodules/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation

# ==============================
# weight
# ==============================
hf auth login
# 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
uv pip install 'huggingface-hub[cli]<1.0'
TAG=hf
hf download --repo-type model --local-dir checkpoints/${TAG}-download --max-workers 1 facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}


# ==============================
# running
# ==============================
python demo_our.py
```

MIDI

```Bash
# ===============================
# 安装环境
# ===============================
uv venv --python 3.10
source .venv/bin/activate
sudo apt update
sudo apt install -y libgl1-mesa-glx libglib2.0-0

uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install setuptools==65.5.0 wheel==0.38.4
uv pip install -r requirements.txt --no-build-isolation
uv pip install git+https://github.com/huanngzh/MV-Adapter --no-build-isolation
uv pip install "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" --no-build-isolation

# 纹理生成
uv pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
uv pip install jaxtyping typeguard --no-build-isolation
uv pip install pymeshlab==2022.2.post3
uv pip install gltflib


# ==============================
# running
# ==============================
python -m scripts.inference_midi --rgb assets/example_data/Cartoon-Style/00_rgb.png --seg assets/example_data/Cartoon-Style/00_seg.png --output-dir "./"

# 纹理生成：
python /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-thirdparts/MIDI-3D/scripts/image_to_textured_scene_our.py \
  --rgb /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-thirdparts/data/image.png \
  --mask-dir /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-thirdparts/data/case_mask/image_0000_start30 \
  --seed 42 \
  --num-inference-steps 30 \
  --guidance-scale 7.0 \
  --do-image-padding \
  --output-dir /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-thirdparts/data/textured_output

# 几何生成
python /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-thirdparts/MIDI-3D/scripts/inference_midi_our.py \
  --rgb /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-thirdparts/data/image.png \
  --mask-dir /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-thirdparts/data/case_mask/image_0000_start30 \
  --seed 42 \
  --num-inference-steps 50 \
  --guidance-scale 7.0 \
  --do-image-padding \
  --output-dir /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-thirdparts/data/midi_output
```

**装配**

运行脚本：

```Bash
# MIDI
python /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-thirdparts/MIDI-3D/scripts/image_to_textured_scene_our.py \
  --rgb /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-thirdparts/data/image.png \
  --mask-dir /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-thirdparts/data/case_mask/image_0000_start30 \
  --seed 42 \
  --num-inference-steps 30 \
  --guidance-scale 7.0 \
  --do-image-padding \
  --output-dir /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-thirdparts/MIDI-3D/outputs/textured_output

# SAM3D
```

### 2.5 资产扩充

桌面检测脚本，通过 yolo-seg 26 检测图片中是否存在桌面，且其余资产大于阈值

```Bash
代码文件夹在：/mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-embodiedintelligence-AssertsProcess/TableVerse_post
代码在：/mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-embodiedintelligence-AssertsProcess/TableVerse_post/demo/detect_yoloe26.py
uv环境在：/mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-embodiedintelligence-AssertsProcess/TableVerse_post/yoloE
在新机器上可能需要：/mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-embodiedintelligence-AssertsProcess/TableVerse_post/demo/install_env.sh 才能用uv环境。

运行指令：
python demo/detect_yoloe26.py \
--log-file /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-embodiedintelligence-AssertsProcess/TableVerse_post/log_file/v1_frame_200_debug.log \
--output-dir /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/temp/
```

## 三、第三人称VLA数据

### 3.1 主线数据处理：

1. How2100M 数据
    1. 数据粗筛

      数据粗筛阶段主要是利用 How2100M 自带的标注信息对原始 youtube 视频分段，并利用 mediapipe 去检测这一片段视频中手部出现的占比 $R_{hand}(\%)=\frac{N_{hand}}{N_{total}}\times 100\%$ ，只保留手部出现比例超过 90% 的视频段，并命名为 {video_name}/clip_{id}.mp4。例如：
    
    ![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=NDhkOTUyNTBiNGZiZDkyNWI0OTc4ZjZmY2M3OWM1MGRfeU1pV3N6cGpLVzcwUWJyY0RoNDRSZzFCcGpXRVVyY1JfVG9rZW46T2FQMmJGQjYyb0xaZWJ4UHNkR2NMblhrbmhlXzE3ODI5ODIyNTg6MTc4Mjk4NTg1OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

    > ⚠️ 注意：
    > 
    > - how2100M信息在一个json中：/mnt/bn/isp-traindata-lf3/datasets/howto100m/caption.json
    >     
    > - Youtube 视频名称 {video_name} 是一个11位长度的名称，有时候从 tos 拿到的链接没有mp4后缀，可以考虑直接加上mp4。例如：http://tosv.byted.org/obj/isp-light-asset-cn/youtube_video/PcoMQqQ5aj0_best 和 http://tosv.byted.org/obj/isp-light-asset-cn/youtube_video_3170/cfFit9jdJGc_1_1_137+251.mp4 都可能是我们能拿到的 url 链接，直接用最后一级的前11位作为 video_name 并加上 .mp4 就可以
    >     
    > - caption 分段信息其实是分段的开始和结束时间，我是按照这个视频所有的分段数来设定 clip_{id}.mp4 的，每一个clip 都用 mediapipe 检测，如果检测的手部出现的帧 / 总的检测帧大于90%，就保留这个 clip
    >     
    > - 如果后续回捞的话，可以考虑在这一块放低标准后，和已有的 clip_{id} 通过这个 id 去重后回捞到新的视频片段

      粗筛代码仓库在：
    
    https://code.byted.org/aidp/algo-embodiedintelligence-vlacoarse/tree/main/ray_process?ref_type=heads
    
      为了速度，代码主要适配了 Ray 用于分布式处理，Ray 的使用可以参考：[基于Ray做分布式训练数据处理](https://bytedance.larkoffice.com/docx/Cqwnd95TpodzNhxTyXVc3u4JnOd?from=from_copylink)。从文档中可以看到如何创建 ray 的任务实例，如何做分发。

    2. 数据精筛

      数据位置总览：[VLA数据记录](https://bytedance.larkoffice.com/wiki/S4u2w8agSiOFCqk0EeYc9KDCned?from=from_copylink)。How2100M是我们第三人称项目处理的第一个数据集，是分批次处理的，总共分了 7 批，每一批的数量不太相同，主要是按照某一个时间点去划分的（确定一个时间点往前的算作一批），由于前期探索阶段，导致每一批数据的文件位置、存储格式、文件命名有不少差异，这些不同都要在精筛阶段适配。代码主要在：
    
    https://code.byted.org/aidp/algo-embodiedintelligence-vlafineprocess/blob/xutao_dev/thing_ray/third_ray_end_version_batch_end0519.sh?ref_type=heads
    
      其中有很多过滤逻辑，每一个逻辑过滤率如下图所示：
    
    ![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=N2YxYzY5MzM4ZGQyOWM0ZmE1OWU3NmJjZGJiNjhiYjlfQ3JwaHNTdnNwMFdkMXBqR3NMOWZFYndFVVpDVHc5TWJfVG9rZW46UVg0emJoNzZmb2hBWW94dTYxN2NBSFNKbmdmXzE3ODI5ODIyNTg6MTc4Mjk4NTg1OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)
    
      筛选完之后主要是合成训练数据，训练数据格式可以参考：/mnt/bn/embodied-lf/public/mano_vis/README.md

2. 新YouTube 100w数据
3. 新YouTube 500w数据
    1. 过滤代码：
        
        https://code.byted.org/aidp/algo-embodiedintelligence-vlafineprocess/blob/xutao_dev/scripts/0_New_500w_Ytb_Filter_By_Title.py?ref_type=heads
        
            运行：
        
        ```Bash
        python 0_New_500w_Ytb_Filter_By_Title.py \
            --input /mnt/hdfs/byte_isp_data/user/sun.ny/data/video_500w/part5.csv \
            --output /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-embodiedintelligence-vlafineprocess/outputs/new_500w_ytb/part5_filter.csv
        ```
        
    2. 下载代码：
        
        https://code.byted.org/aidp/algo-embodiedintelligence-vlafineprocess/blob/xutao_dev/scripts/1_Download_from_tos_HDFS.py?ref_type=heads
        
            运行：
        
        ```Bash
        ray job submit \
          --address="http://[fdbd:dc01:25:620::215]:10247" \
          --no-wait -- python3 /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-embodiedintelligence-vlafineprocess/scripts/1_Download_from_tos_HDFS.py \
          --csv-path /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-embodiedintelligence-vlafineprocess/outputs/new_500w_ytb/part1_filter.csv \
            /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-embodiedintelligence-vlafineprocess/outputs/new_500w_ytb/part2_filter.csv \
            /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-embodiedintelligence-vlafineprocess/outputs/new_500w_ytb/part3_filter.csv \
            /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-embodiedintelligence-vlafineprocess/outputs/new_500w_ytb/part4_filter.csv \
            /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-embodiedintelligence-vlafineprocess/outputs/new_500w_ytb/part5_filter.csv  \
          --output-path hdfs://haruna/home/byte_isp_data/user/xuexutao/new_500w_ytb/video \
          --num-workers 300 \
          --use-ray
        ```

总共有：4284922 个Youtube视频，位置在：hdfs://haruna/home/byte_isp_data/user/xuexutao/new_500w_ytb/video

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=ODZiYzE0NjkwYjM5YWVhZDUyYjZiZjM3NWFmYTkwYWRfQkR2Rnk0MDRMZUxsRDVHWHdKZDVvZkoyT21jR2FlaE1fVG9rZW46VHQ0NmJuYzhzb0hXRWR4NVVGc2N4SzdKbnNlXzE3ODI5ODIyNTg6MTc4Mjk4NTg1OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

处理进度位于：[【06.22】youtube-500W数据记录](https://bytedance.larkoffice.com/wiki/RHaqwcqeLifk8dk9zvhcXTKdnX3?from=from_copylink)，其中 **过滤下载** 部分可以看到每一个csv过滤的比例，约15%

### 3.2 模型优化

**1、ActionFilter**

**训练**

ActionFilter是一个视频二分类模型，基于冻结的 Vjepa 编码器对视频进行编码，并通过训练分类头得到分类模型

对于数据和训练方式可以参考如下文档：[Actionfilter训练脚本](https://bytedance.larkoffice.com/wiki/ZJhQw6ZzTi4eBikGsQccjKjbn7f?from=from_copylink)，模型训练完后可以得到一个 latest.pt 权重文件。

代码仓库在：https://code.byted.org/aidp/algo-embodiedintelligence-fastvideo/tree/xutao_dev?ref_type=heads

**推理**

对于训练完的模型可以通代码推理外，还可以使用线上 MIPP 由@李乾 创建的算子推理，算子链接在：

https://cloud.bytedance.net/mipp/operators/operator_details?_clu=default&_idc=lf&domainId=aidp&name=aidp%2Faidp_fast_video&tab=basic&x-bc-region-id=bytedance&x-bc-vdc=http&x-bc-vregion=China-North&x-resource-account=public

输入测试视频可以参考我的1000测试样本：hdfs://haruna/home/byte_isp_data/user/xuexutao/action_filter_finetune_data/test_1000.parquet

MIPP推理逻辑使用滑窗逻辑，通过检测 128 frame 长度的视频输出 0/1 标签，换算到 30fps 约为 4.27s。

由于MIPP无法做合并操作，所以得到的结果是parquet中，每一行对应一个jsonl文件，后续可以合并在一个文件中进行可视化～

**可视化**

推理完的结果可以参考我的可视化代码，输入采用的是MIPP输出的jsonl(合并成一个文件)：

eg:hdfs://haruna/home/byte_isp_data/user/xuexutao/action_filter_finetune_data/mipp_output/ActionFilter_videos_new.jsonl

https://code.byted.org/aidp/algo-embodiedintelligence-vlafineprocess/blob/xutao_dev/ActionFilter/vis_online/run.sh?ref_type=heads

```Bash
#!/bin/bash

# ====================== 前置准备 ======================
# 需要使用 1.0.0.20 版本的 toscli 工具
# 安装命令如下：
# VER=1.0.0.20
# wget https://luban-source.byted.org/repository/scm/toutiao.tos.toscli_${VER}.tar.gz -O toscli_${VER}.tar.gz
# rm -rf output && mkdir output && tar zxvf toscli_${VER}.tar.gz -C output && rm toscli_${VER}.tar.gz && mv output ${VER}
# cd ${VER}
# ./toscli 
# 
# 可以在当前目录下的到 1.0.0.20 的文件夹，那就是成功了～
# ======================================================




set -euo pipefail

# ====================== 配置参数 ======================
JSONL_FILE="ActionFilter_videos_new.jsonl"                          # 输入的 JSONL 文件名，可以是 action filter 的 mipp 算子的到的 jsonl 结果，合成的到的 jsonl
HTML_FILE="ActionFilter_1000_videos_0624.html"                      # 生成的 HTML 文件名
TOS_VIDEO_PREFIX="xuexutao/action_filter_vis/260623_1000video_vis"  # TOS 远端视频存储路径
TOS_HTML_PREFIX="xuexutao/action_filter_vis"                        # TOS 远端 HTML 存储路径
WORKER_VIDEO=8
WORKER_HTML=1
# ======================================================

echo "===== Step1:上传 JSONL 内所有视频至 TOS ====="
python for_visual_upload_mp4.py \
    --jsonl_path "${JSONL_FILE}" \
    --remote_prefix "${TOS_VIDEO_PREFIX}" \
    --num_workers "${WORKER_VIDEO}"

echo "===== Step2:基于 JSONL 生成可视化 HTML ====="
python for_visual_generate_html.py \
    --jsonl "${JSONL_FILE}" \
    --output-html "${HTML_FILE}"

echo "===== Step3:上传生成的 HTML 文件至 TOS ====="
python for_visual_upload_mp4.py \
    --input_path "./${HTML_FILE}" \
    --remote_prefix "${TOS_HTML_PREFIX}" \
    --num_workers "${WORKER_HTML}"

echo -e "\n==================== 任务全部完成 ===================="
echo "可视化页面访问地址：https://tosv.byted.org/obj/aidp-embodied-ai-inspect/${TOS_HTML_PREFIX}/${HTML_FILE}"
```

参考前端界面：

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=MjNlODk2MTZkNzJlMDE4MjEzMTYzYTAwZWE5OTgzMDZfUXBsWXR5bll6MEwzTHZhOFhVTnVrNXFyZk5Sa3F1aG5fVG9rZW46SEV1VGJoSHI3b2VxRjF4MnllYmM3ZmNUbjM4XzE3ODI5ODIyNTg6MTc4Mjk4NTg1OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

2、验证下游VLA模型

**代码**

https://code.byted.org/aidp-playground/algo-embodiedintelligence-gigabrain/tree/origin_dev?ref_type=heads

采用的 giga_brain_0 代码框架，环境安装脚本如下：

```Bash
# nvcc -V
echo "================================"
echo "install cuda driver......"
echo "================================"
cd ~
# 倾向于使用已有的 .cache 有些模型可以不用下载
rm -rf ~/.cache/
sudo ln -s /mnt/bn/aidp-data-3d-lf1/xxt/.cache/ ~/.cache
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
sudo apt-get install tmux -y

echo "================================"
echo "install env......"
echo "================================"
export HTTP_PROXY=http://bj-rd-proxy.byted.org:8118
export http_proxy=http://bj-rd-proxy.byted.org:8118
export https_proxy=http://bj-rd-proxy.byted.org:8118
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "================================"
echo "install env......"
echo "================================"
mkdir -p ~/miniconda3/envs/giga_brain_0
tar -zxvf /opt/tiger/algo-embodiedintelligence-gigabrain/pretrained_models/giga_brain_0.tar.gz -C ~/miniconda3/envs/giga_brain_0 --strip-components=1
sudo apt update
sudo apt install -y libgl1-mesa-glx libglib2.0-0
source ~/miniconda3/bin/activate giga_brain_0
echo "================================"
echo "Installation successful"
echo "================================\n\n"

#### 开始训练
echo "================================"
echo "start train......"
echo "================================"
cd /opt/tiger/algo-embodiedintelligence-gigabrain
# 训练配置
DATA_PATHS="/mnt/bn/aidp-data-3d-lf1/vla_data_outer/train_data/exoweb_outer_part0/episodic_annotations" \
NORM_STATS_PATH='/mnt/bn/aidp-data-3d-lf1/vla_data_outer/train_data/exoweb_outer_part0/norm_stats_134.json' \
OUTPUT_PATH='/mnt/bn/aidp-data-3d-lf1/vla_data_outer/train_experiments/outer_data_ex1/giga_brain_0_web_episode_outer_data_134_binning_dual_expert_18k/'
# 多卡训练
GIGA_GPU_IDS=0,1,2,3,4,5,6,7  python scripts/train.py --config configs.giga_brain_0_web_episode_from_scratch.config

# 训练完后等待，也可省略，训练完后直接停止任务
while true; do sleep 86400; done
```

可以直接复制以下任务实例创建任务：

https://ml.bytedance.net/development/instance/jobs/4e9bad495bbfe296?tabState=run_info&trialId=361762556

**数据**

https://code.byted.org/aidp-playground/algo-embodiedintelligence-gigabrain/blob/origin_dev/data_process/run.sh?ref_type=heads

代码的主要逻辑是，通过 ray 读入 hdf5 数据，并且将数据解码开后，转换为第三人称训练可用的数据格式

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=OGI2NWU2MmI4ZDk0N2NmZGEzNzQyNzVkNDFmMDExMmZfQTJkNTBmZnRweEhxblZPN3NUeGZ2c2xKV3htTVR4NDhfVG9rZW46Sm1NdGJvc1Zxb3Zhc3B4MnIwd2NJZ3hYbjlkXzE3ODI5ODIyNTg6MTc4Mjk4NTg1OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

**运行**

https://code.byted.org/aidp-playground/algo-embodiedintelligence-gigabrain/blob/origin_dev/run_step.sh?ref_type=heads

```Bash
# =====================================
# 1、计算 Norm 统计值
# =====================================
OMP_NUM_THREADS=1 ray job submit \
 --runtime-env-json='{
    "pip": [
      "numpydantic",
      "tyro",
      "diffusers==0.36.0",
      "decord"
    ]
  }' \
 --address="http://[2605:340:cdb1:130:411a:deb3:f822:7c61]:10720" --no-wait --working-dir ./scripts -- python3 compute_norm_stats_ray.py \
    --data-paths /mnt/bn/aidp-data-3d-lf1/vla_data_outer/train_data/exoweb_outer_part0/episodic_annotations \
    --output-path /mnt/bn/aidp-data-3d-lf1/vla_data_outer/train_data/exoweb_outer_part0/norm_stats_134.json \
    --embodiment-id 3 \
    --delta-mask True \
    --sample-rate 1.0 \
    --action-chunk 50 \
    --action-dim 134 \
    --dataset-class-name WebEpisodeDataset \
    --num-workers 300



# =====================================
# 2、训练模型
# =====================================
EPISODE_FRAME_INDEX_FILE="/mnt/bn/aidp-data-3d-lf1/vla_data_outer/train_data/exoweb_outer_part0/episode_frame_index.npz" \
DATA_PATHS="/mnt/bn/aidp-data-3d-lf1/vla_data_outer/train_data/exoweb_outer_part0/episodic_annotations" \
NORM_STATS_PATH="/mnt/bn/aidp-data-3d-lf1/vla_data_outer/train_data/exoweb_outer_part0/norm_stats_134_2.json" \
OUTPUT_PATH="/mnt/bn/aidp-data-3d-lf1/vla_data_outer/train_experiments/debug2/giga_brain_0_web_episode_outer_data_134_binning_dual_expert_18k_2/" \
GIGA_GPU_IDS=0,1,2,3,4,5,6,7 \
python scripts/train.py --config configs.giga_brain_0_web_episode_from_scratch.config



# =====================================
# 3、模型推理
# =====================================
data_path='/mnt/bn/aidp-data-3d-lf1/vla_data_outer/train_data/exoweb_outer_part0/'
json_output_path='/mnt/bn/aidp-data-3d-lf1/vla_data_outer/train_data/exoweb_outer_part0/norm_stats_134_2.json'
output_path='/mnt/bn/aidp-data-3d-lf1/vla_data_outer/validation/debug2/giga_brain_0_web_episode_outer_data_134_binning_dual_expert_18k_2/checkpoint_epoch_1_step_18000'
python scripts/inference_web_episode.py \
    --model-path /mnt/bn/aidp-data-3d-lf1/vla_data_outer/train_experiments/debug2/giga_brain_0_web_episode_outer_data_134_binning_dual_expert_18k_2/models/checkpoint_epoch_1_step_18000/model/ \
    --data-path $data_path \
    --norm-stats-path $json_output_path \
    --tokenizer-model-path /opt/tiger/algo-embodiedintelligence-gigabrain/pretrained_models/models--google--paligemma-3b-pt-224 \
    --embodiment-id 3 \
    --original-action-dim 134 \
    --action-chunk 50 \
    --num-eval-samples 8 \
    --sample-stride 100 \
    --device cuda:0 \
    --output-path $output_path \
    --use-expert2 \
    --present-img-keys observation.images.cam_high 
    # 和训练对齐就好，训练用三张图片，推理也用三张，训练用一张，推理就用一张
```

### 3.3 其他需求

1、convert caption

需要通过调用 Batch Job 将结构化语言刷成自然语言，后续可以适配其他任务，代码在：

https://code.byted.org/aidp-playground/algo-embodiedintelligence-AssertsProcess/blob/d8547ff34699488de68cacf5f8d4ef85813a571a/upload_tos_ray/run.sh?ref_type=commits#L30:1-30:4

代码主要读入一批训练数据，并通过将文本转为 BatchJob 的输入，后续直接提交给 batchJob 即可实现

2、pipeline34.sh

新链路中处理 ActionFilter 输出结果的 ray 进程：

```Bash
cd /mnt/bn/embodied-lf/qian/internet_clipping/internet_video_process
ray job submit --address="http://[2605:340:cdb1:130:411a:deb3:f822:7c61]:10036" --no-wait --working-dir . -- python3.9 pipeline_34.py \
    --transnet_out_dir /mnt/hdfs/byte_isp_data/user/liqian/new_500w_ytb/mipp_transnetv2/results_001 \
    --scenes_out_dir "/mnt/hdfs/byte_isp_data/user/liqian/new_500w_ytb/mipp_transnetv2/scenes_out_001" \
    --output_parquet "/mnt/hdfs/byte_isp_data/user/liqian/new_500w_ytb/mipp_actionfilter/scenes_001.jsonl.parquet" \
    --num_workers 64 \
    --local_prefix /mnt/hdfs/byte_isp_data/user \
    --hdfs_prefix hdfs://haruna/home/byte_isp_data/user
```

## 四、世界模型质检

### 4.1 多Voter投票机制

代码仓库：

https://code.byted.org/aidp/algo-multimodal-worldmodel/tree/xutao_dev?ref_type=heads

运行命令：

```Bash
# 运行
INPUT="/mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-multimodal-worldmodel/world_action/neeko_anno/0624_500.jsonl" \
OUTPUT="/mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-multimodal-worldmodel/world_action_output_online/0624_blind_vote_500" \
FPS=4 \
MAX_FRAMES=32 \
ENTITY_CROP=true \
EVENT_CONCURRENCY=10 \
VIDEO_CONCURRENCY=2 \
bash world_action/run.sh --referee 21lite

# 可视化
python -m world_action.visual_html.visual_world_action_blind_vote \
    --summary $OUTPUT \
    --output-html /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-multimodal-worldmodel/world_action/visual_html/visual_500.html
```

### 4.2 视频调用

代码在：world_action_one_video_one_call

一个视频根据world action 分成的片段，每个world action片段采用 5fps 的方式拼接成大图并配合 这个区间的 object 和 机标caption 生成新的 qc_caption

```Bash
# 运行
python world_action_one_video_one_call.call_one_process.py \
    --input /mnt/bn/embodied-lf/qian/algo-multimodal-worldmodel/evaluate/results/machine_37/selected_5cases_with_video_path.jsonl \
    --output /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-multimodal-worldmodel/world_action_one_video_one_call_output/online_5_v1 \
    --model-preset DOUBAO_2_1_LITE_MLLM \
    --fps 5 \
    --max-frames-per-event 80 \
    --dump-prompt


# 可视化
python -m  world_action_one_video_one_call.visual_html_generate \
    --input-dir /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-multimodal-worldmodel/world_action_one_video_one_call_output/online_5_v1 \
    --output-html /mnt/bn/aidp-data-3d-lf1/xxt/merlin/workspace/algo-multimodal-worldmodel/world_action_one_video_one_call/visual_output/visual_one_call_qc_v1.html
```
