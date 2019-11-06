import os
import unittest
from sphinx.application import Sphinx


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
        app = Sphinx(self.source_dir,
                     self.config_dir,
                     self.output_dir,
                     self.doctree_dir,
                     buildername='html',
                     warningiserror=False,
                     )
        app.build(force_all=self.all_files)
        # TODO: additional checks here if needed

    # def test_text_documentation(self):
    #     The same, but with different buildername
        # app = Sphinx(self.source_dir,
        #              self.config_dir,
        #              self.output_dir,
        #              self.doctree_dir,
        #              buildername='latex',
        #              warningiserror=False,
        #              )
        # app.build(force_all=self.all_files)
        # TODO:  additional checks if needed

    def tearDown(self):
        # TODO: clean up the output directory
        pass


