"""
server/hier_fedper.py  –  Hier-FedPer 边缘服务器（阶段三 CCSF，D-113 / D-114）

edge 层与 Hier-FedRep 完全相同：只对 body 索引做样本加权聚合（有防御走 robust_mean），私有层位置不参与。
body 索引本来就含 BN 的 moving 统计量 → 统计量随 body 聚合；FedPer 客户端不把它们私有化（client/hier_fedper.py），
于是与官方 FedPer「buffers 跟着 base」一致。区别只在客户端的训练（每步联合训练全部变量）。
"""

from .hier_fedrep import HierFedRepEdgeServer


class HierFedPerEdgeServer(HierFedRepEdgeServer):

    _method_label = "Hier-FedPer"
