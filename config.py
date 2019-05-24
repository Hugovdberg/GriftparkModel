# setup absolute paths to the executables based on the OS

import sys
import os

bindir = "./bin"
exeext = ""

if sys.platform.lower() == "darwin":
    exepth = os.path.join(bindir, "mac")
elif sys.platform.lower() == "linux":
    exepth = os.path.join(bindir, "linux")
elif "win" in sys.platform.lower():
    exeext = ".exe"
    is_64bits = sys.maxsize > 2 ** 32
    if is_64bits:
        winarch = "win64"
    else:
        winarchpltfrm = "win32"
    exepth = os.path.join(bindir, winarch)
else:
    raise Exception("Could not find binaries for {}".format(sys.platform))


def add_to_config(name, exeext):
    exe = os.path.abspath(os.path.join(exepth, name.format(exeext)))
    return exe


def get_path(exe):
    return os.path.abspath(os.path.join(exepth, "{}{}".format(exe, exeext)))


mfexe = get_path("mf2005")
mpexe = get_path("mp7")
mtexe = get_path("mt3dms")
mf6exe = get_path("mf6")
mtusgsexe = get_path("mt3dusgs")

exelist = [mfexe, mpexe, mtexe, mf6exe, mtusgsexe]
for e in exelist:
    if not os.path.isfile(e):
        print("Executable file could not be found: {}".format(e))
    else:
        print("Executable file found: {}".format(e))
