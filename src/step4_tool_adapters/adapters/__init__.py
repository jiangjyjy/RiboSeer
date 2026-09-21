"""Concrete adapters — one per tool of the MAESTRO library (paper Table 1).

Local execution (RELAY runs the tool):

  - boltz2_adapter      (Cat A, conda env: boltz)
  - chai1_adapter       (Cat A, conda env: chai1, single-sequence)
  - rf2na_adapter       (Cat A, conda env: RF2NA2, single-sequence)
  - rfaa_adapter        (Cat A, conda env: RFAA, single-sequence)
  - p2rank_adapter      (Cat B, Java, no conda env)
  - fpocket_adapter     (Cat B, geometric, system binary)
  - deeppocket_adapter  (Cat B, conda env: DeepPocket, GPU)
  - equipnas_adapter    (Cat C, conda env: EquiPNAS, GPU)
  - nucleicnet_adapter  (Cat C, in-process against a local checkout, GPU;
                         not thread-safe — run with parallel_workers: 1)
  - graphbind_adapter   (Cat C, conda env: GraphBind)
  - haddock3_adapter    (Cat D, conda env: haddock3)
  - hdock_adapter       (Cat D, HDOCKlite binaries, CPU)

Manual web submission (the adapter writes the submission payload and reads
back the results you downloaded; see the README):

  - alphafold3_adapter   (Cat A, AlphaFold Server)
  - rnabindrplus_adapter (Cat C, RNABindRPlus server, results by email)
  - bindup_adapter       (Cat C, BindUP server, batch or single form)
"""
from .alphafold3_adapter import AlphaFold3Adapter
from .bindup_adapter import BindUPAdapter
from .boltz2_adapter import Boltz2Adapter
from .chai1_adapter import Chai1Adapter
from .deeppocket_adapter import DeepPocketAdapter
from .equipnas_adapter import EquiPNASAdapter
from .fpocket_adapter import FpocketAdapter
from .graphbind_adapter import GraphBindAdapter
from .haddock3_adapter import Haddock3Adapter
from .hdock_adapter import HdockAdapter
from .nucleicnet_adapter import NucleicNetAdapter
from .p2rank_adapter import P2RankAdapter
from .rf2na_adapter import RF2NAAdapter
from .rfaa_adapter import RFAAAdapter
from .rnabindrplus_adapter import RNABindRPlusAdapter

__all__ = [
    "AlphaFold3Adapter", "BindUPAdapter", "Boltz2Adapter", "Chai1Adapter",
    "DeepPocketAdapter", "EquiPNASAdapter", "FpocketAdapter",
    "GraphBindAdapter", "Haddock3Adapter", "HdockAdapter",
    "NucleicNetAdapter", "P2RankAdapter", "RF2NAAdapter", "RFAAAdapter",
    "RNABindRPlusAdapter",
]
