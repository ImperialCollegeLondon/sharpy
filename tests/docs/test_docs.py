import unittest
import os

# Skipped to avoid dependency of minimal environment with sphinx
# we agreed that it is best to have the core of SHARPy only in the tests
# ADC 14 Nov 2019
@unittest.skip
class DocTest(unittest.TestCase):
    """

    """
    route = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/'
    source_dir = route + u'../../docs/source'
    config_dir = route + u'../../docs/source'
    output_dir = route + u'../../docs/build'
    doctree_dir = route + u'../../docs/build/doctrees'
    all_files = 1

    def test_html_documentation(self):
        from sphinx.application import Sphinx
        app = Sphinx(self.source_dir,
                     self.config_dir,
                     self.output_dir,
                     self.doctree_dir,
                     buildername='html',
                     warningiserror=False,
                     )
        app.build(force_all=self.all_files)
        # TODO: additional checks here if needed

    def test_text_documentation(self):
        from sphinx.application import Sphinx
        # The same, but with different buildername
        app = Sphinx(self.source_dir,
                     self.config_dir,
                     self.output_dir,
                     self.doctree_dir,
                     buildername='text',
                     warningiserror=False,
                     )
        app.build(force_all=self.all_files)
        # TODO:  additional checks if needed

    def tearDown(self):
        # TODO: clean up the output directory
        pass


