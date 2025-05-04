from . import communication
from . import datasets
from . import expression
from . import external
from . import gene_score
from . import io
from . import metabolic_task
from . import plotting
from . import preprocessing
from . import reaction_activity
from . import reports
from . import spatial
from . import stats
from .gene_score import (gene_score, compute_gene_scores, compute_gpr_gene_score)
from .metabolic_task import (compute_mt_score)
from .reaction_activity import (compute_reaction_activity)
from .sccellfie_pipeline import (run_sccellfie_pipeline)

__version__ = "0.4.5"