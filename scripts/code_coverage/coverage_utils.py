#! /usr/bin/env python3
# vim: tabstop=4 softtabstop=4

import argparse
from collections import defaultdict
import functools
import jinja2
import json
import logging
import os
import shutil
import sys

# The default name of the html coverage report for a directory.
DIRECTORY_COVERAGE_HTML_REPORT_NAME = "report.html"

# Name of the html index files for different views.
DIRECTORY_VIEW_INDEX_FILE = "directory_view_index.html"
FILE_VIEW_INDEX_FILE = "file_view_index.html"
INDEX_HTML_FILE = "index.html"


def configure_logging(verbose=False, log_file=None):
    """Configures logging settings for later use."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "[%(levelname)s] %(message)s"
    logging.basicConfig(filename=log_file, level=log_level, format=log_format)


def get_coverage_report_root_dir_path(output_dir):
    """The root directory that contains all generated coverage html reports."""
    return os.path.join(output_dir, get_host_platform())


def get_directory_view_path(output_dir):
    """Path to the HTML file for the directory view."""
    return os.path.join(get_coverage_report_root_dir_path(output_dir), DIRECTORY_VIEW_INDEX_FILE)


def get_file_view_path(output_dir):
    """Path to the HTML file for the file view."""
    return os.path.join(get_coverage_report_root_dir_path(output_dir), FILE_VIEW_INDEX_FILE)


def get_html_index_path(output_dir):
    """Path to the main HTML index file."""
    return os.path.join(get_coverage_report_root_dir_path(output_dir), INDEX_HTML_FILE)


def get_full_path(path):
    """Return full absolute path."""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def get_host_platform():
    """Returns the host platform.

    This is separate from the target platform/os that coverage is running for.
    """
    if sys.platform == "win32" or sys.platform == "cygwin":
        return "win"
    if sys.platform.startswith("linux"):
        return "linux"
    else:
        assert sys.platform == "darwin"
        return "mac"


def get_relative_path_to_directory_of_file(target_path, base_path):
    """Returns a target path relative to the directory of base_path.

    This method requires base_path to be a file, otherwise, one should call
    os.path.relpath directly.
    """
    assert os.path.dirname(base_path) != base_path, "Base path '{}' is a directory, please call os.path.relpath directly.".format(base_path)
    base_dir = os.path.dirname(base_path)
    return os.path.relpath(target_path, base_dir)


def merge_two_directories(src_dir_path, dst_dir_path):
    """Merge src_dir_path directory into dst_path directory."""
    for filename in os.listdir(src_dir_path):
        dst_path = os.path.join(dst_dir_path, filename)
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        os.rename(os.path.join(src_dir_path, filename), dst_path)
    shutil.rmtree(src_dir_path)


def WriteRedirectHtmlFile(from_html_path, to_html_path):
    """Writes a html file that redirects to another html file."""
    to_html_relative_path = get_relative_path_to_directory_of_file(to_html_path, from_html_path)
    content = """\
<!DOCTYPE html>
<html>
  <head>
    <!-- HTML meta refresh URL redirection -->
    <meta http-equiv="refresh" content="0; url={}">
  </head>
</html>""".format(to_html_relative_path)

    with open(from_html_path, "w") as f:
        f.write(content)


class CoverageSummary:
    """Encapsulates coverage summary representation."""

    def __init__(self,
                 regions_total=0,
                 regions_covered=0,
                 functions_total=0,
                 functions_covered=0,
                 lines_total=0,
                 lines_covered=0):
        """Initializes CoverageSummary object."""
        self._summary = {
            "regions": {
                "total": regions_total,
                "covered": regions_covered
            },
            "functions": {
                "total": functions_total,
                "covered": functions_covered
            },
            "lines": {
                "total": lines_total,
                "covered": lines_covered
            }
        }

    def get(self):
        """Returns summary as a dictionary."""
        return self._summary

    def add_summary(self, other_summary):
        """Adds another summary to this one element-wise."""
        for feature in self._summary:
            self._summary[feature]["total"] += other_summary.get()[feature]["total"]
            self._summary[feature]["covered"] += other_summary.get()[feature]["covered"]


class CoverageReportHtmlGenerator:
    """Encapsulates coverage html report generation.

    The generated html has a table that contains links to other coverage reports.
    """

    def __init__(self, output_dir, output_path, table_entry_type):
        """
        Args:
            output_dir: Path to the dir for writing coverage report to.
            output_path: Path to the html report that will be generated.
            table_entry_type: Type of the table entries to be displayed in the table
                              header. For example: "Path".
        """
        css_file_name = "style.css"
        css_absolute_path = os.path.join(output_dir, css_file_name)
        assert os.path.exists(css_absolute_path), (
            "css file doesn\'t exit. Please make sure `llvm-cov show -format=html` "
            "is called first, and the css file is generated at '{}'.".format(
            css_absolute_path))

        self._css_absolute_path = css_absolute_path
        self._output_dir = output_dir
        self._output_path = output_path
        self._table_entry_type = table_entry_type

        self._table_entries = []
        self._total_entry = {}

        source_dir = os.path.dirname(os.path.realpath(__file__))
        template_dir = os.path.join(source_dir, "html_templates")

        jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir),
                                       trim_blocks=True)
        self._header_template = jinja_env.get_template("header.html")
        self._table_template = jinja_env.get_template("table.html")
        self._footer_template = jinja_env.get_template("footer.html")

        css_path = os.path.join(source_dir, "static", "css", "style.css")
        self._style_overrides = open(css_path).read()

    def add_link_to_another_report(self, html_report_path, name, summary):
        """Adds a link to another html report in this report.

        The link to be added is assumed to be an entry in this directory.
        """
        # Use relative paths instead of absolute paths to make the generated reports
        # portable.
        html_report_relative_path = get_relative_path_to_directory_of_file(
            html_report_path, self._output_path)

        table_entry = self._create_table_entry_from_coverage_summary(
            summary, html_report_relative_path, name,
            os.path.basename(html_report_path) ==
            DIRECTORY_COVERAGE_HTML_REPORT_NAME)
        self._table_entries.append(table_entry)

    def create_totals_entry(self, summary):
        """Creates an entry corresponds to the "Totals" row in the html report."""
        self._total_entry = self._create_table_entry_from_coverage_summary(summary)

    def _create_table_entry_from_coverage_summary(self, summary, href=None, name=None, is_dir=None):
        """Creates an entry to display in the html report."""
        assert (href is None and name is None and is_dir is None) or (
            href is not None and name is not None and is_dir is not None), (
                "The only scenario when href or name or is_dir can be None is when "
                "creating an entry for the Totals row, and in that case, all three "
                "attributes must be None.")

        entry = {}
        if href is not None:
            entry["href"] = href
        if name is not None:
            entry["name"] = name
        if is_dir is not None:
            entry["is_dir"] = is_dir

        summary_dict = summary.get()
        for feature in summary_dict:
            if summary_dict[feature]["total"] == 0:
                percentage = 0.0
            else:
                percentage = float(summary_dict[feature]["covered"]) / summary_dict[feature]["total"] * 100

            color_class = self._get_color_class(percentage)
            entry[feature] = {
                "total": summary_dict[feature]["total"],
                "covered": summary_dict[feature]["covered"],
                "percentage": "{:6.2f}".format(percentage),
                "color_class": color_class
            }

        return entry

    def _get_color_class(self, percentage):
        """Returns the css color class based on coverage percentage."""
        if percentage >= 0 and percentage < 80:
            return "red"
        if percentage >= 80 and percentage < 100:
            return "yellow"
        if percentage == 100:
            return "green"
        raise ValueError("Invalid coverage percentage: {}".format(percentage))

    def write_html_coverage_report(self, no_file_view):
        """Writes html coverage report.

        In the report, sub-directories are displayed before files and within each
        category, entries are sorted alphabetically.
        """

        def entry_cmp(left, right):
            """Compare function for table entries."""
            if left["is_dir"] != right["is_dir"]:
                return -1 if left["is_dir"] == True else 1
            return -1 if left["name"] < right["name"] else 1

        self._table_entries = sorted(self._table_entries, key=functools.cmp_to_key(entry_cmp))

        css_path = os.path.join(self._output_dir, "style.css")

        directory_view_path = get_directory_view_path(self._output_dir)
        directory_view_href = get_relative_path_to_directory_of_file(directory_view_path, self._output_path)

        # File view is optional in the report.
        file_view_href = None
        if not no_file_view:
            file_view_path = get_file_view_path(self._output_dir)
            file_view_href = get_relative_path_to_directory_of_file(file_view_path, self._output_path)

        html_header = self._header_template.render(
            css_path=get_relative_path_to_directory_of_file(css_path, self._output_path),
            directory_view_href=directory_view_href,
            file_view_href=file_view_href,
            style_overrides=self._style_overrides)

        html_table = self._table_template.render(
            entries=self._table_entries,
            total_entry=self._total_entry,
            table_entry_type=self._table_entry_type)

        html_footer = self._footer_template.render()

        with open(self._output_path, "w") as html_file:
            html_file.write(html_header + html_table + html_footer)


class CoverageReportPostProcessor:
    """Post processing of code coverage reports produced by llvm-cov."""

    def __init__(self, output_dir, src_root_dir, summary_data, no_file_view, path_equivalence=None):
        # Caller provided parameters.
        self.output_dir = output_dir
        self.src_root_dir = os.path.normpath(get_full_path(src_root_dir))
        if not self.src_root_dir.endswith(os.sep):
            self.src_root_dir += os.sep
        self.summary_data = json.loads(summary_data)
        assert len(self.summary_data["data"]) == 1
        self.no_file_view = no_file_view

        # The root directory that contains all generated coverage html reports.
        self.report_root_dir = get_coverage_report_root_dir_path(self.output_dir)

        # Path to the HTML file for the directory view.
        self.directory_view_path = get_directory_view_path(self.output_dir)

        # Path to the HTML file for the file view.
        self.file_view_path = get_file_view_path(self.output_dir)

        # Path to the main HTML index file.
        self.html_index_path = get_html_index_path(self.output_dir)

        self.path_map = None
        if path_equivalence:
            def _prepare_path(path):
                path = os.path.normpath(path)
                if not path.endswith(os.sep):
                    # A normalized path does not end with '/', unless it is a root dir.
                    path += os.sep
                return path

            self.path_map = [_prepare_path(p) for p in path_equivalence.split(",")]
            assert len(self.path_map) == 2, "Path equivalence argument is incorrect."

    def _map_to_local(self, path):
        """Maps a path from the coverage data to a local path."""
        if not self.path_map:
            return path
        return path.replace(self.path_map[0], self.path_map[1], 1)

    def calculate_per_directory_coverage_summary(self, per_file_coverage_summary):
        """Calculates per directory coverage summary."""
        logging.debug("Calculating per-directory coverage summary.")
        per_directory_coverage_summary = defaultdict(lambda: CoverageSummary())

        for file_path in per_file_coverage_summary:
            summary = per_file_coverage_summary[file_path]
            parent_dir = os.path.dirname(file_path)

            while True:
                per_directory_coverage_summary[parent_dir].add_summary(summary)
                if os.path.normpath(parent_dir) == os.path.normpath(self.src_root_dir):
                    break
                parent_dir = os.path.dirname(parent_dir)

        logging.debug("Finished calculating per-directory coverage summary.")
        return per_directory_coverage_summary

    def get_coverage_html_report_path_for_directory(self, dir_path):
        """Given a directory path, returns the corresponding html report path."""
        assert os.path.isdir(self._map_to_local(dir_path)), "'{}' is not a directory.".format(dir_path)
        html_report_path = os.path.join(get_full_path(dir_path), DIRECTORY_COVERAGE_HTML_REPORT_NAME)

        # '+' is used instead of os.path.join because both of them are absolute
        # paths and os.path.join ignores the first path.
        return self.report_root_dir + html_report_path

    def get_coverage_html_report_path_for_file(self, file_path):
        """Given a file path, returns the corresponding html report path."""
        assert os.path.isfile(self._map_to_local(file_path)), "'{}' is not a file.".format(file_path)
        html_report_path = os.extsep.join([get_full_path(file_path), "html"])

        # '+' is used instead of os.path.join because both of them are absolute
        # paths and os.path.join ignores the first path.
        return self.report_root_dir + html_report_path

    def generate_file_view_html_index_file(self, per_file_coverage_summary, file_view_index_file_path):
        """Generates html index file for file view."""
        logging.debug("Generating file view html index file as '{}'.".format(file_view_index_file_path))
        html_generator = CoverageReportHtmlGenerator(self.output_dir, file_view_index_file_path, "Path")
        totals_coverage_summary = CoverageSummary()

        for file_path in per_file_coverage_summary:
            totals_coverage_summary.add_summary(per_file_coverage_summary[file_path])
            html_generator.add_link_to_another_report(
                self.get_coverage_html_report_path_for_file(file_path),
                os.path.relpath(file_path, self.src_root_dir),
                per_file_coverage_summary[file_path])

        html_generator.create_totals_entry(totals_coverage_summary)
        html_generator.write_html_coverage_report(self.no_file_view)
        logging.debug("Finished generating file view html index file.")

    def generate_per_file_coverage_summary(self):
        """Generate per file coverage summary using coverage data in JSON format."""
        files_coverage_data = self.summary_data["data"][0]["files"]

        per_file_coverage_summary = {}
        for file_coverage_data in files_coverage_data:
            file_path = file_coverage_data["filename"]
            # skip files which are outside the source root (e.g. unit tests themselves)
            if not file_path.startswith(self.src_root_dir):
                continue

            summary = file_coverage_data["summary"]
            if summary["lines"]["count"] == 0:
                continue

            per_file_coverage_summary[file_path] = CoverageSummary(
                regions_total=summary["regions"]["count"],
                regions_covered=summary["regions"]["covered"],
                functions_total=summary["functions"]["count"],
                functions_covered=summary["functions"]["covered"],
                lines_total=summary["lines"]["count"],
                lines_covered=summary["lines"]["covered"])

        logging.debug("Finished generating per-file code coverage summary.")
        return per_file_coverage_summary

    def generate_per_directory_coverage_in_html(self, per_directory_coverage_summary, per_file_coverage_summary):
        """Generates per directory coverage breakdown in html."""
        logging.debug("Writing per-directory coverage html reports.")
        for dir_path in per_directory_coverage_summary:
            self.generate_coverage_in_html_for_directory(dir_path, per_directory_coverage_summary, per_file_coverage_summary)
        logging.debug("Finished writing per-directory coverage html reports.")

    def generate_coverage_in_html_for_directory(self, dir_path, per_directory_coverage_summary, per_file_coverage_summary):
        """Generates coverage html report for a single directory."""
        html_generator = CoverageReportHtmlGenerator(
            self.output_dir, self.get_coverage_html_report_path_for_directory(dir_path),
            "Path")

        for entry_name in os.listdir(self._map_to_local(dir_path)):
            entry_path = os.path.normpath(os.path.join(dir_path, entry_name))

            if entry_path in per_file_coverage_summary:
                entry_html_report_path = self.get_coverage_html_report_path_for_file(entry_path)
                entry_coverage_summary = per_file_coverage_summary[entry_path]
            elif entry_path in per_directory_coverage_summary:
                entry_html_report_path = self.get_coverage_html_report_path_for_directory(entry_path)
                entry_coverage_summary = per_directory_coverage_summary[entry_path]
            else:
                # Any file without executable lines shouldn't be included into the
                # report. For example, OWNER and README.md files.
                continue

            html_generator.add_link_to_another_report(entry_html_report_path,
                                                      os.path.basename(entry_path),
                                                      entry_coverage_summary)

        html_generator.create_totals_entry(per_directory_coverage_summary[dir_path])
        html_generator.write_html_coverage_report(self.no_file_view)

    def generate_directory_view_html_index_file(self):
        """Generates the html index file for directory view.

        Note that the index file is already generated under src_root_dir, so this
        file simply redirects to it, and the reason of this extra layer is for
        structural consistency with other views.
        """
        directory_view_index_file_path = self.directory_view_path
        logging.debug("Generating directory view html index file as '{}'.".format(
                      directory_view_index_file_path))
        src_root_html_report_path = self.get_coverage_html_report_path_for_directory(self.src_root_dir)
        WriteRedirectHtmlFile(directory_view_index_file_path, src_root_html_report_path)
        logging.debug("Finished generating directory view html index file.")

    def rename_default_coverage_directory(self):
        """Rename the default coverage directory into platform specific name."""
        # llvm-cov creates "coverage" subdir in the output dir. We would like to use
        # the platform name instead, as it simplifies the report dir structure when
        # the same report is generated for different platforms.
        default_report_subdir_path = os.path.join(self.output_dir, "coverage")
        if not os.path.exists(default_report_subdir_path):
            logging.error("Default coverage report dir does not exist: {}.".format(default_report_subdir_path))

        if not os.path.exists(self.report_root_dir):
            os.mkdir(self.report_root_dir)

        merge_two_directories(default_report_subdir_path, self.report_root_dir)

    def overwrite_html_reports_index_file(self):
        """Overwrites the root index file to redirect to the default view."""
        html_index_file_path = self.html_index_path
        directory_view_index_file_path = self.directory_view_path
        WriteRedirectHtmlFile(html_index_file_path, directory_view_index_file_path)

    def clean_up_output_dir(self):
        """Perform a cleanup of the output dir."""
        # Remove the default index.html file produced by llvm-cov.
        index_path = os.path.join(self.output_dir, INDEX_HTML_FILE)
        if os.path.exists(index_path):
            os.remove(index_path)

    def prepare_html_report(self):
        self.rename_default_coverage_directory()

        per_file_coverage_summary = self.generate_per_file_coverage_summary()

        if not self.no_file_view:
            self.generate_file_view_html_index_file(per_file_coverage_summary, self.file_view_path)

        per_directory_coverage_summary = self.calculate_per_directory_coverage_summary(per_file_coverage_summary)

        self.generate_per_directory_coverage_in_html(per_directory_coverage_summary, per_file_coverage_summary)
        self.generate_directory_view_html_index_file()

        # The default index file is generated only for the list of source files,
        # needs to overwrite it to display per directory coverage view by default.
        self.overwrite_html_reports_index_file()
        self.clean_up_output_dir()

        html_index_file_path = "file://" + get_full_path(self.html_index_path)
        logging.info("Index file for html report is generated as '{}'.".format(html_index_file_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("coverage_utils", description="Code coverage utils.")
    parser.add_argument("-v", "--verbose", action="store_true",
          help="Prints additional debug output.")
    parser.add_argument("--output-dir",
          help="Path to the report dir.", required=True)
    parser.add_argument("--src-root-dir",
          help="Path to the src root dir.", required=True)
    parser.add_argument("--summary-file",
          help="Path to the summary file.", required=True)
    parser.add_argument("--path-equivalence",
          help="Map the paths in the coverage data to local source files path (=<from>,<to>)")

    args = parser.parse_args()
    configure_logging(args.verbose)

    with open(args.summary_file) as f:
        summary_data = f.read()

    processor = CoverageReportPostProcessor(
        args.output_dir,
        args.src_root_dir,
        summary_data,
        no_file_view=False,
        path_equivalence=args.path_equivalence)
    processor.prepare_html_report()
