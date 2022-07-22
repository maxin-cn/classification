#
#  Modified by liujunjie on 2020/12/17.
#  Copyright (c) Meituan. Holding Limited
#  Email liujunjie10@meituan.com
#

__version__ = '999.0.0-developing'


from SKDX.runtime.env_vars import dispatcher_env_vars
from SKDX.tools.utils import ClassArgsValidator

if dispatcher_env_vars.SDK_PROCESS != 'dispatcher':
    from SKDX.tools.trial import *
    from SKDX.tools.smartparam import *
    from SKDX.common.nas_utils import training_update

class NoMoreTrialError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo
