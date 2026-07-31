import pytorch_lightning as L


class MDLM_NLP(L.LightningModule): # masked diffusion language model, with Explorative Modeling via learned per-candidate mask token embeddings
    def __init__(self, hparams):
        super().__init__()
        raise ValueError("MDLM code is coming soon, it is not available yet in this release")
