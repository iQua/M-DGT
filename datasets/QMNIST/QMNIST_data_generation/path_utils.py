#!/usr/bin/env python
# -*- coding: utf-8 -*-


''' python inherent libs '''
import os


''' third parts libs '''


''' custom libs '''


# The child path can be any type
def path_compatible_system(base_path, added_chind_path):
    pattern_win = re.compile('/')
    pattern_lx = re.compile('\\\\')

    win_child_path = re.sub(pattern_win, '\\\\', added_chind_path)
    lx_child_path = re.sub(pattern_lx, '/', added_chind_path)

    if "\\" in base_path and not "/" in base_path:
        # Windows path
        compat_path = os.path.join(base_path, win_child_path)

    if "/" in base_path and not "\\" in base_path:
        # linux path
        compat_path = os.path.join(base_path, lx_child_path)

    return compat_path

def get_common_path():
    # get the current path
    cur_path = os.getcwd()
    # parent's path for current path
    cur_project_path = os.path.abspath(os.path.dirname(cur_path)
                                +os.path.sep+".")
    # The previous two levels of the current file
    main_dir = os.path.abspath(os.path.dirname(cur_project_path)+os.path.sep+"..")

    sources_path = os.path.join(main_dir, "sources")

    return main_dir, sources_path, cur_project_path

MAIN_dir_p, SOURCES_p, CUR_PROJECT_p = get_common_path()
