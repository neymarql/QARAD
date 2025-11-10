import torch, torchvision.transforms.functional as TF
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()

@torch.no_grad()
def clip_similarity(img_BCHW, prompt_list):
    pil_imgs = [TF.to_pil_image(img) if torch.is_tensor(img) else Image.open(img) for img in img_BCHW]
    ipt = _proc(text=prompt_list, images=pil_imgs, return_tensors="pt", truncation=True,
                         max_length=77, padding=True).to("cuda")
    out = _model(**ipt)
    return torch.cosine_similarity(out.image_embeds, out.text_embeds)
