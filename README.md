# model-template
when we use this model template
we can easilly serve the model through a sample command like this


using this sample command

```docker
docker run  -e GIT_REPO_URL=https://the_address -e GIT_USER=your_user -e GIT_TOKEN=**** -e MODEL_CONFIG=the_chosen_config -P <host_port>:<container_port>  registry.ibagher.ir/tritonserver:25.01-pt
```

---

the `on_start.sh` and `on_stop.sh` will be excecuted on start and stop of the docker container
we can put a bash script to do anything we want in them

---

the `requirements.txt` will be installed in the docker file


we need to pull the docker before, of course


instead of this readme, in the original repo we will put the sample usage code

somthing like this:


```python 

import tritonclient.http as httpclient
import numpy as np

TRITON_URL = "the_url"
MODEL = "model_name"

def infer_with_model(image_path):
	client = httpcoient.InferenceServerClient(url=TRITON_URL)
	... and so on

```
