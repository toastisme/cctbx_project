

import re
import os

RAYONIX_DIR = "/global/project/projectdirs/lcls/asmit/for_derek/rayonix_files_with_ts_refined_on_JF1M"

def get_tstamp(name):
    s = re.search("201805[0-9]+", name)
    return name[s.start(): s.end()]
