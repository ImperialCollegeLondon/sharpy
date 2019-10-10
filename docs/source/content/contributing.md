# Contributing to SHARPy

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
    
    - Once you are done, run the following ``SHARPy`` command:
    ```bash
    sharpy any_string -d
    ```
    
    - If you are making minor updates to docstrings (i.e. you are not documenting a previously undocumented
    function/class/module) you can simply change directory to  `sharpy/docs` and run 
    ```bash
    make html
    ```
    
    - Your documentation will compile and warnings will appear etc. You can check the result by opening
    ```bash
    docs/build/index.html
    ```
    and navigating to your recently created page.
    
    - Make sure that **before committing** any changes in the documentation you update the entire ``docs`` directory
    by running
    ```bash
    sharpy any_string -d
    ```
    
Thank you for reading through this and contributing to make SHARPy a better documented, more user friendly code!