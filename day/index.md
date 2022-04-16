---
layout: main
main: true
title: Day
---


<div class="loading-animation">
<!-- <CENTER> <iframe width="560" height="315" src="https://www.youtube.com/embed/9YAB9jMzy7k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> </CENTER> -->
<!-- <CENTER> <video width="260" height="215" src = "../assets/couple.MOV"  autoplay muted loop playsinline></video> </CENTER> -->

    {% include hashtag.html %}
    <ul class="catalogue">
        {% assign sorted = site.pages | sort: 'date' | reverse | where: 'type', 'day' %}
        {% for page in sorted %}
        {% include post-list.html %}
        {% endfor %}
    </ul>

</div>
