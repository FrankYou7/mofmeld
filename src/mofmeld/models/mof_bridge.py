import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


class BridgeBlock(nn.Module):
    """
    A single bridge block with optional cross-attention followed by self-attention
    and a feed-forward MLP.
    """

    def __init__(self, dim: int, nhead: int, hidden_dim: int, cross_attn: bool = True):
        super().__init__()
        self.cross = cross_attn

        if self.cross:
            self.xattn = nn.MultiheadAttention(dim, nhead, batch_first=True)

        self.sattn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            q: Query tokens of shape [B, Nq, D]
            kv: Structure-conditioned key/value tensor of shape [B, Nk, D]

        Returns:
            Updated query tokens of shape [B, Nq, D]
        """
        if self.cross and kv is not None:
            q = q + self.xattn(self.norm1(q), self.norm1(kv), self.norm1(kv))[0]

        q = q + self.sattn(self.norm2(q), self.norm2(q), self.norm2(q))[0]
        q = q + self.mlp(self.norm3(q))
        return q


class MOFBridgeModel(nn.Module):
    """
    Q-Former-style bridge module for mapping a fixed-size structure embedding
    into a set of LLM-compatible query tokens.

    Architecture:
        64 -> 768 -> alternating cross/self-attention -> 768 -> 4096
    """

    def __init__(
        self,
        input_dim: int = 64,
        proj_dim: int = 768,
        llama_hidden: int = 4096,
        num_query: int = 32,
        num_layers: int = 8,
        nhead: int = 8,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim)
        self.query = nn.Parameter(torch.randn(1, num_query, proj_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, num_query, proj_dim))

        self.blocks = nn.ModuleList(
            BridgeBlock(
                dim=proj_dim,
                nhead=nhead,
                hidden_dim=hidden_dim,
                cross_attn=(i % 2 == 0),
            )
            for i in range(num_layers)
        )

        self.norm = nn.LayerNorm(proj_dim)
        self.to_llama = nn.Linear(proj_dim, llama_hidden)

    def forward(self, struct_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            struct_vec: Structure embedding tensor of shape [B, 64]

        Returns:
            Query-token embeddings in the LLM hidden space of shape [B, 32, 4096]
        """
        x = self.proj(struct_vec)  # [B, 768]
        q = self.query.expand(x.size(0), -1, -1) + self.pos_emb  # [B, 32, 768]
        kv = x.unsqueeze(1)  # [B, 1, 768]

        for i, blk in enumerate(self.blocks):
            q = blk(q, kv if i % 2 == 0 else None)

        q = self.norm(q)
        return self.to_llama(q)  # [B, 32, 4096]


class MOFMultiModal(nn.Module):
    """
    Unified multimodal model used across:
    - Stage-I pretraining
    - Stage-II fine-tuning
    - Downstream inference
    - Multimodal embedding export

    Components:
    - MOFBridgeModel
    - Frozen LLM backbone
    - Optional matching head
    - Learnable logit scale for contrastive learning
    """

    def __init__(self, bridge_ckpt: str | None = None, llama_path: str | None = None):
        super().__init__()

        if llama_path is None:
            raise ValueError("`llama_path` must be provided when initializing MOFMultiModal.")

        self.bridge = MOFBridgeModel()

        if bridge_ckpt:
            state_dict = torch.load(bridge_ckpt, map_location="cpu")
            # Support both plain state dict and checkpoint dict with a "model" key
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            self.bridge.load_state_dict(state_dict)

        self.tokenizer = AutoTokenizer.from_pretrained(
            llama_path,
            trust_remote_code=True,
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            llama_path,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
        )

        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad = False

        self.match_head = nn.Linear(4096 * 2, 1)
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )

    def forward_pred(
        self,
        struct_vec: torch.Tensor,
        text_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stage-I pretraining objective: structure-conditioned generation loss.
        """
        query_emb = self.bridge(struct_vec)                    # [B, 32, 4096]
        txt_emb = self.llm.model.embed_tokens(text_ids)       # [B, T, 4096]
        inputs = torch.cat([query_emb, txt_emb], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs,
            labels=labels,
        )
        return outputs.loss

    def forward_corr(
        self,
        struct_vec: torch.Tensor,
        text_cls_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stage-I pretraining objective: structure-text contrastive loss (InfoNCE).
        """
        q = self.bridge(struct_vec).mean(1)   # [B, 4096]
        t = text_cls_emb                      # [B, 4096]

        q = F.normalize(q, dim=-1)
        t = F.normalize(t, dim=-1)

        logits = (q @ t.t()) * self.logit_scale.exp()
        targets = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, targets)

    def forward_match(
        self,
        struct_vec: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stage-I pretraining objective: structure-text matching logit.
        """
        q = self.bridge(struct_vec).mean(1)   # [B, 4096]
        x = torch.cat([q, text_emb], dim=-1)  # [B, 8192]
        logit = self.match_head(x).squeeze(-1)
        return logit

    def forward_finetune(
        self,
        struct_vec: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stage-II fine-tuning objective.

        Args:
            struct_vec: [B, 64]
            input_ids: [B, L]
            attention_mask: [B, L]
            labels: [B, L]

        Returns:
            Language-modeling loss over the answer tokens.
        """
        query_emb = self.bridge(struct_vec)                   # [B, 32, 4096]
        txt_emb = self.llm.model.embed_tokens(input_ids)      # [B, L, 4096]
        full_embeds = torch.cat([query_emb, txt_emb], dim=1)  # [B, 32+L, 4096]

        bridge_mask = torch.ones(
            (attention_mask.shape[0], query_emb.shape[1]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([bridge_mask, attention_mask], dim=1)

        bridge_labels = torch.full(
            (labels.shape[0], query_emb.shape[1]),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        full_labels = torch.cat([bridge_labels, labels], dim=1)

        outputs = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )
        return outputs.loss

    def forward_embedding(
        self,
        struct_vec: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Export pooled multimodal structure-guided embeddings for retrieval or analysis.

        Args:
            struct_vec: [B, 64]
            input_ids: [B, L]

        Returns:
            Pooled embedding of shape [B, 4096]
        """
        query_emb = self.bridge(struct_vec)                   # [B, 32, 4096]
        _ = self.llm.model.embed_tokens(input_ids)            # kept for interface consistency
        pooled = query_emb.mean(1)                            # [B, 4096]
        return pooled

    def forward(
        self,
        struct_vec: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Default forward path used in stage-II fine-tuning and DDP training.
        """
        return self.forward_finetune(struct_vec, input_ids, attention_mask, labels)