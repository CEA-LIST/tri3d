{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
    {% if objtype == "class"%}:inherited-members:{% endif %}