{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block properties%}
   {% if attributes %}
   .. rubric:: {{ _('Properties') }}

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {%- if item.startswith('wo') %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
