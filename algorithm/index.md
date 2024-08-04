---
layout: main
main: true
title: Algorithm
---

<div class="loading-animation">

{% include hashtag.html %}
<ul class="catalogue">
    {% assign sorted = site.pages | sort: 'date' | reverse | where: 'type', 'algorithm' %}
    {% for page in sorted %}
    {% include post-list.html %}
    {% endfor %}
</ul>
</div>
