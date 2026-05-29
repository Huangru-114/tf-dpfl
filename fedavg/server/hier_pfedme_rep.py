"""
server/hier_pfedme_rep.py  –  Hier-pFedMe-Rep 边缘服务器

按两套 Hier-Rep 共享的 edge 层设计，edge 聚合逻辑与 Hier-Ditto-Rep **完全相同**：
仅对 backbone 索引做样本加权聚合，并用 Ditto 近端把 edge backbone 拉向 cloud 的
全局 backbone φ*（系数 mu_edge，φ* 在 global round 内固定）：

    φ_e_bb ← mean_bb − mu_edge · (φ_e_bb_prev − φ*_bb)

两套方法的区别只在 **client backbone 个性化机制**（此处对应的 client 用 pFedMe
Moreau envelope），edge 层无差异。因此本类直接继承 HierDittoRepEdgeServer，
只改日志标签，避免重复实现。
"""

from .hier_ditto_rep import HierDittoRepEdgeServer


class HierPFedMeRepEdgeServer(HierDittoRepEdgeServer):

    _method_label = "Hier-pFedMe-Rep"
