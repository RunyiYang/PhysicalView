"""Prompted edit adapter using the available Qwen 2511 checkpoint."""
import os
from pathlib import Path


def main():
    from agents.edit import inpaint_qwen
    def load_qwen():
        import json
        import torch
        import diffusers
        from diffusers import QwenImageEditPlusPipeline
        root = Path(os.environ.get('PHIVIEW_QWEN_MODEL',
            '/group/worldcept/hf_cache/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9'))
        pipe = QwenImageEditPlusPipeline.from_pretrained(root, torch_dtype=torch.bfloat16,
                                                        local_files_only=True)
        if not getattr(pipe.transformer.config, 'zero_cond_t', False):
            raise RuntimeError('Qwen 2511 requires a Diffusers build with zero_cond_t support')
        pipe.enable_model_cpu_offload()
        (inpaint_qwen.C.OUT/'inpaint'/'model-receipt.json').write_text(json.dumps({
            'model': str(root), 'diffusers': diffusers.__version__, 'zero_cond_t': True,
            'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()}, indent=2))
        print(f'[PhiView] Prompt model: {root}', flush=True)
        return pipe
    inpaint_qwen.load_qwen = load_qwen
    inpaint_qwen.main()


if __name__ == '__main__':
    main()
