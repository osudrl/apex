import gym

def print_logo(subtitle=""):
  print("                   AAA                                                                           ")
  print("                  A:::A                                                                          ")
  print("                 A:::::A                                                                         ")
  print("                A:::::::A                                                                        ")
  print("               A:::::::::A           ppppp   ppppppppp       eeeeeeeeeeee    xxxxxxx      xxxxxxx")
  print("              A:::::A:::::A          p::::ppp:::::::::p    ee::::::::::::ee   x:::::x    x:::::x ")
  print("             A:::::A A:::::A         p:::::::::::::::::p  e::::::eeeee:::::ee  x:::::x  x:::::x  ")
  print("            A:::::A   A:::::A        pp::::::ppppp::::::pe::::::e     e:::::e   x:::::xx:::::x   ")
  print("           A:::::A     A:::::A        p:::::p     p:::::pe:::::::eeeee::::::e    x::::::::::x    ")
  print("          A:::::AAAAAAAAA:::::A       p:::::p     p:::::pe:::::::::::::::::e      x::::::::x     ")
  print("         A:::::::::::::::::::::A      p:::::p     p:::::pe::::::eeeeeeeeeee       x::::::::x     ")
  print("        A:::::AAAAAAAAAAAAA:::::A     p:::::p    p::::::pe:::::::e               x::::::::::x    ")
  print("       A:::::A             A:::::A    p:::::ppppp:::::::pe::::::::e             x:::::xx:::::x   ")
  print("      A:::::A               A:::::A   p::::::::::::::::p  e::::::::eeeeeeee    x:::::x  x:::::x  ")
  print("     A:::::A                 A:::::A  p::::::::::::::pp    ee:::::::::::::e   x:::::x    x:::::x ")
  print("    AAAAAAA                   AAAAAAA p::::::pppppppp        eeeeeeeeeeeeee  xxxxxxx      xxxxxxx")
  print("                                      p:::::p                                                    ")
  print("                                      p:::::p                                                    ")
  print("                                      p:::::::p                                                  ")
  print("                                      p:::::::p                                                  ")
  print("                                      p:::::::p                                                  ")
  print("                                      ppppppppp                                                  ")
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

