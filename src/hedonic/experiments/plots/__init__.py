"""Paper / experiment plotting utilities (optional: matplotlib, seaborn).

Public entrypoint for the PHYSA V1020 disjoint synthetic figures::

    from hedonic.experiments.plots.paper_figures import main, generate_all_figures

CLI::

    hedonic-exp plots --help
    hedonic-exp plots --smoke --output_dir /tmp/figs
    hedonic-exp plots --data .../resultados.csv.gzip --output_dir .../figures
"""

from __future__ import annotations

from hedonic.experiments.plots.paper_figures import (
    FIGURE_NAMES,
    generate_all_figures,
    main,
)

__all__ = ["FIGURE_NAMES", "generate_all_figures", "main"]
