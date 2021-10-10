---
layout: main
main: true
title: dayily
---

<div class="loading-animation">
    {% include hashtag.html %}
    <ul class="catalogue">
        {% assign sorted = site.pages | sort: 'date' | reverse | where: 'type', 'day' %}
        {% for page in sorted %}
        {% include post-list.html %}
        {% endfor %}
    </ul>
</div>
