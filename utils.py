from collections import namedtuple  # Tuple with fixed names

import numpy as np

StressPeriod = namedtuple(
    typename='StressPeriod',
    field_names=('period_length', 'n_steps',
                 'step_multiplier', 'steady_state'),
    defaults=(float(1), int(1), float(1), True)
)

stress_period_dtype = np.dtype([('period_length', np.float),
                                ('n_steps', np.int),
                                ('step_multiplier', np.float),
                                ('steady_state', np.bool)])
