import gym

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m\033[48m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
   L1 = '\033[38;5;194m'
   L2 = '\033[38;5;193m'
   L3 = '\033[38;5;192m'
   L4 = '\033[38;5;191m'
   L5 = '\033[38;5;198m'
   L6 = '\033[38;5;198m'
   L7 = '\033[38;5;197m'
   L8 = '\033[38;5;214m'
   L9 = '\033[38;5;202m'
   L10 = '\033[38;5;240m'

"""
   \e[38;5;190m




\e[38;5;192m
\e[38;5;193m
\e[38;5;194m
\e[38;5;195m
\e[38;5;196m
"""


def print_logo(subtitle="", option=2):
  classy = True
  if not classy:
    print()
    print(color.BOLD + color.L9 + "          :::     :::::::::  :::::::::: :::    ::: " + color.END)
    print(color.BOLD + color.L9 + "       :+: :+:   :+:    :+: :+:        :+:    :+: " + color.END)
    print(color.BOLD + color.L9 + "     +:+   +:+  +:+    +:+ +:+         +:+  +:+ " + color.END)
    print(color.BOLD + color.L9 + "   +#++:++#++: +#++:++#+  +#++:++#     +#++:+ " + color.END)
    print(color.BOLD + color.L10 + "  +#+     +#+ +#+        +#+         +#+  +#+ " + color.END)
    print(color.BOLD + color.L10 + " #+#     #+# #+#        #+#        #+#    #+# " + color.END)
    print(color.BOLD + color.L10 + color.UNDERLINE    + "###     ### ###        ########## ###    ### " + color.END)
  else:
    if option == 1:
      print(color.BOLD + color.L9 +  "         .8.         " + color.L10 + " 8 888888888o   " + color.L9 + "8 8888888888   `8.`8888.      ,8' ")
      print(color.BOLD + color.L9 +  "        .888.        " + color.L10 + " 8 8888    `88. " + color.L9 + "8 8888          `8.`8888.    ,8' ")
      print(color.BOLD + color.L9 +  "       :88888.       " + color.L10 + " 8 8888     `88 " + color.L9 + "8 8888           `8.`8888.  ,8' ")
      print(color.BOLD + color.L9 +  "      . `88888.      " + color.L10 + " 8 8888     ,88 " + color.L9 + "8 8888            `8.`8888.,8' ")
      print(color.BOLD + color.L9 +  "     .8. `88888.     " + color.L10 + " 8 8888.   ,88' " + color.L9 + "8 888888888888     `8.`88888' ")
      print(color.BOLD + color.L10 + "    .8`8. `88888.    " + color.L9  + " 8 888888888P'  " + color.L10 + "8 8888             .88.`8888. ")
      print(color.BOLD + color.L10 + "   .8' `8. `88888.   " + color.L9  + " 8 8888         " + color.L10 + "8 8888            .8'`8.`8888. ")
      print(color.BOLD + color.L10 + "  .8'   `8. `88888.  " + color.L9  + " 8 8888         " + color.L10 + "8 8888           .8'  `8.`8888. ")
      print(color.BOLD + color.L10 + " .888888888. `88888. " + color.L9  + " 8 8888         " + color.L10 + "8 8888          .8'    `8.`8888. ")
      print(color.BOLD + color.L10 + ".8'       `8. `88888." + color.L9  + " 8 8888         " + color.L10 + "8 888888888888 .8'      `8.`8888. " + color.END)
    if option == 2:
      print(color.BOLD + color.L9 +  "         .8.         " + color.L10 + " 8 888888888o   " + color.L9 + "8 8888888888   `8.`8888.      ,8' ")
      print(color.BOLD + color.L9 +  "        .888.        " + color.L10 + " 8 8888    `88. " + color.L9 + "8 8888          `8.`8888.    ,8' ")
      print(color.BOLD + color.L9 +  "       :88888.       " + color.L10 + " 8 8888     `88 " + color.L9 + "8 8888           `8.`8888.  ,8' ")
      print(color.BOLD + color.L9 +  "      . `88888.      " + color.L10 + " 8 8888     ,88 " + color.L9 + "8 8888            `8.`8888.,8' ")
      print(color.BOLD + color.L9 +  "     .8. `88888.     " + color.L10 + " 8 8888.   ,88' " + color.L9 + "8 888888888888     `8.`88888' ")
      print(color.BOLD + color.L9 + "    .8`8. `88888.    " + color.L10  + " 8 888888888P'  " + color.L9 + "8 8888             .88.`8888. ")
      print(color.BOLD + color.L9 + "   .8' `8. `88888.   " + color.L10  + " 8 8888         " + color.L9 + "8 8888            .8'`8.`8888. ")
      print(color.BOLD + color.L9 + "  .8'   `8. `88888.  " + color.L10  + " 8 8888         " + color.L9 + "8 8888           .8'  `8.`8888. ")
      print(color.BOLD + color.L9 + " .888888888. `88888. " + color.L10  + " 8 8888         " + color.L9 + "8 8888          .8'    `8.`8888. ")
      print(color.BOLD + color.L9 + ".8'       `8. `88888." + color.L10  + " 8 8888         " + color.L9 + "8 888888888888 .8'      `8.`8888. " + color.END)
    if option == 3:
      print(color.BOLD + color.L9 +  "         .8.         " + color.L10 + " 8 888888888o   " + color.L9 + "8 8888888888   `8.`8888.      ,8' ")
      print(color.BOLD + color.L9 +  "        .888.        " + color.L10 + " 8 8888    `88. " + color.L9 + "8 8888          `8.`8888.    ,8' ")
      print(color.BOLD + color.L9 +  "       :88888.       " + color.L10 + " 8 8888     `88 " + color.L9 + "8 8888           `8.`8888.  ,8' ")
      print(color.BOLD + color.L9 +  "      . `88888.      " + color.L10 + " 8 8888     ,88 " + color.L9 + "8 8888            `8.`8888.,8' ")
      print(color.BOLD + color.L9 +  "     .8. `88888.     " + color.L10 + " 8 8888.   ,88' " + color.L9 + "8 888888888888     `8.`88888' ")
      print(color.BOLD + color.L10 + "    .8`8. `88888.    " + color.L9  + " 8 888888888P'  " + color.L10 + "8 8888             .88.`8888. ")
      print(color.BOLD + color.L10 + "   .8' `8. `88888.   " + color.L9  + " 8 8888         " + color.L10 + "8 8888            .8'`8.`8888. ")
      print(color.BOLD + color.L10 + "  .8'   `8. `88888.  " + color.L9  + " 8 8888         " + color.L10 + "8 8888           .8'  `8.`8888. ")
      print(color.BOLD + color.L10 + " .888888888. `88888. " + color.L9  + " 8 8888         " + color.L10 + "8 8888          .8'    `8.`8888. ")
      print(color.BOLD + color.L10 + ".8'       `8. `88888." + color.L9  + " 8 8888         " + color.L10 + "8 888888888888 .8'      `8.`8888. " + color.END)
  print("\n")
  print(subtitle)
  print("\n")

def gym_factory(path, **kwargs):
    from functools import partial

    """
    This is (mostly) equivalent to gym.make(), but it returns an *uninstantiated* 
    environment constructor.

    Since environments containing cpointers (e.g. Mujoco envs) can't be serialized, 
    this allows us to pass their constructors to Ray remote functions instead 
    (since the gym registry isn't shared across ray subprocesses we can't simply 
    pass gym.make() either)

    Note: env.unwrapped.spec is never set, if that matters for some reason.
    """
    spec = gym.envs.registry.spec(path)
    _kwargs = spec._kwargs.copy()
    _kwargs.update(kwargs)

    try:
      if callable(spec._entry_point):
        cls = spec._entry_point(**_kwargs)
      else:
        cls = gym.envs.registration.load(spec._entry_point)
    except AttributeError:
      if callable(spec.entry_point):
        cls = spec.entry_point(**_kwargs)
      else:
        cls = gym.envs.registration.load(spec.entry_point)

    return partial(cls, **_kwargs)

