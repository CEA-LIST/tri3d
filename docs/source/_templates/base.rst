{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
    {% if objtype == "class"%}:inherited-members:{% endif %}
    {% if objtype == "class"%}:special-members: __matmul__, __len__{% endif %}
   