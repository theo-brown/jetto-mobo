Installation
============

Python environment
------------------
Install torch? CUDA? hdf5 library?

Then:
::
    pip install jetto_mobo

JETTO/JINTRAC
-------------

To build JINTRAC Singularity image from a compressed Docker image (``.tgz``), use:
::

    singularity build <singularity-image-name>.sif docker-archive://<docker-image-name>.tgz


Development
-----------
For development:
::
    pip install jetto_mobo[dev]

For contributing guidelines, see `Contributing`_.
