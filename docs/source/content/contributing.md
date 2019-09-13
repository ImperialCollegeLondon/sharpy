# Contributing to SHARPy
## Bug fixes and features

SHARPy is a collaborative effort, and this means that some coding practices need
to be encouraged so that the code is kept tidy and consistent. Any user is welcome
to raise issues for bug fixes and feature proposals through Github.

If you are submitting a bug report:

1. Make sure your SHARPy, xbeam and uvlm local copies are up to date and in the
same branch.

2. Double check that your python distribution is updated by comparing with 
the `utils/environment_*.yml` file.

3. Try to assemble a minimal working example that can be run quickly and easily.

4. Describe as accurately as possible your setup (OS, path, compilers...) and
the problem.

5. Raise an issue with all this information in the Github repo and label it
`potential bug`.

Please bear in mind that we do not have the resources to provide support for
user modifications of the code through Github. If you have doubts about how to modify certain
parts of the code, contact us through email and we will help you as much as we can.

If you are fixing a bug:

1. THANKS!

2. Please create a pull request from your modified fork, and describe in a few
lines which bug you are fixing, a minimal example that triggers the bug and how you
are fixing it. We will review it ASAP and hopefully it will be incorporated in the
code!

If you have an idea for new functionality but do not know how to implement it:

1. We welcome tips and suggestions from users, as it allow us to broaden the scope
of the code. The more people using it, the better!

2. Feel free to fill an issue in Github, and tag it as `feature proposal`. Please
understand that the more complete the description of the potential feature, the more
likely it is that some of the developers will give it a go.

If you have developed new functionality and you want to share it with the world:

1. AWESOME! Please follow the same instructions than for the bug fix submission.
If you have some peer-reviewed references related to the new code, even better, as
it will save us some precious time.

## Code formatting

We try to follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) standards
(with spaces, no tabs please!) and [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html).
We do not ask you to freak out over formatting, but please, try to keep it tidy and
descriptive. A good tip is to run `pylint` [https://www.pylint.org/](https://www.pylint.org/)
to make sure there are no obvious formatting problems.




## Documentation

Contributing to SHARPy's documentation benefits everyone. As a developer, writing documentation helps you better 
understand what you have done and whether your functions etc make logical sense. As a user, any documentation is better 
than digging through the code. The more we have documented, the easier the code is to use and the more users we can 
have.

If you want to contribute by documenting code, you have come to the right place. 

SHARPy is documented using Sphinx and it extracts the documentation directly from the source code. It is then sorted
into directories automatically and a human readable website generated. The amount of work you need to do is minimal. 
That said, the recipe for a successfully documented class, function, module is the following:

1. Your documentation has to be written in ReStructuredText (rst). I know, another language... hence I will leave
    a few tips:
    
    - Inline code is written using two backticks ` `` `
    
    - Inline math is written as ``:math:`1+\exp^{i\pi} = 0` ``. Don't forget the backticks!
        
    - Math in a single or multiple lines is simple:
        
        ```rst
            .. math:: 1 + \exp{i\pi} = 0
        ```    
    
    - Lists in ReStructuredText are tricky, I must admit. Therefore, I will link to some
     [examples](http://docutils.sourceforge.net/docs/user/rst/quickref.html#enumerated-lists). The key resides in not 
     forgetting the spaces, in particular when you go onto a second line!
     
    - The definite example list can be found [here](http://docutils.sourceforge.net/docs/user/rst/quickref.html).
    
2. Titles and docstrings, the bare minimum:       
    - Start docstrings with `r` such that they are interpreted raw:
            
        ```python
        r"""
        My docstring
        """
        ```
    - All functions, modules and classes should be given a title that goes in the first line of the docstring
    
    - If you are writing a whole package with an `__init__.py` file, even if it's empty, give it a human readable
    docstring. This will then be imported into the documentation
    
    - For modules with several functions, the module docstring has to be at the very top of the file, prior to the 
    `import` statements.
    
2. We use the Google documentation style. See [description](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
        
3. Function arguments and returns:
    
    - Function arguments are simple to describe:
        ```python
        def func(arg1, arg2):
        """Summary line.

        Extended description of function.

        Args:
          arg1 (int): Description of arg1
          arg2 (str): Description of arg2

        Returns:
          bool: Description of return value

        """
        return True
       ```

4. Solver settings:

    - If your code has a settings dictionary, with defaults and types then make sure that:
        
        - They are defined as class variables and not instance attributes. 
        
        - Define a `settings_types`, `settings_default` and `settings_description` dictionaries.
        
        - After all your settings, update the docstring with the automatically generated settings table. You will need
        to import the `sharpy.utils.settings` module
        
            ```python
            settings_types = dict()
            settings_default = dict()
            settings_description = dict()

            # keep adding settings
  
            settings_table = sharpy.utils.settings.SettingsTable()
            __doc__ += settings_table.generate(settings_types, settings_default ,settings_description)
            ```

5. See how your docs looks like!
    
    - Once you are done, run the following sharpy command:
    ```bash
    sharpy any_string -d
    ```
    
    - Then go to `sharpy/docs` and run 
    ```bash
    make html
    ```
    
    - Your documentation will compile and warnings will appear etc. You can check the result by opening
    ```bash
    docs/build/index.html
    ```
    and navigating to tour recently created page.
            
Thank you for reading through this and contributing to make SHARPy a better documented, more user friendly code!
