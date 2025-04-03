import os
import scanpy as sc


def _get_file_format(save_name):
    # Common image formats
    image_formats = {'.png', '.pdf', '.svg', '.jpg', '.jpeg', '.tiff'}

    # Check if the save_name ends with any of these formats
    file_ext = os.path.splitext(os.path.basename(save_name))[1].lower()

    if file_ext in image_formats:
        # Return the extension without the dot
        return file_ext[1:]
    else:
        # Return None or default format
        return sc.settings.file_format_figs


def _get_file_dir(save_name):
    dir = os.path.dirname(save_name)
    # Check if path is absolute
    if os.path.isabs(save_name):
        pass
    # Check if user provided a directory
    elif dir != '':
        dir = os.path.abspath(dir)
    else:
        # If not directory provided, combine with scanpy's figdir
        dir = str(sc.settings.figdir.absolute())
    basename = os.path.splitext(os.path.basename(save_name))[0]
    return dir, basename