---
layout: main
main: true
title: About
---
<div class="about">
    <div class="section">
        <div class="title index">01. Career</div>
            <div class="content">
                <ul>
                    Autonomous Driving Software developer<br>
                </ul>
            </div>
    </div>
    <div class="section">
        <div class="title index">02. Cover</div>
            <div class="content">
                <ul>
                    {% assign sorted = site.pages | sort: 'date' | reverse | where: 'type', 'about' %}
                    {% for page in sorted limit: 1%}
                    {% include about-list.html %},
                    {% endfor %}
                    <A href = "https://github.com/SungwookLE">Github</A>
                </ul>
            </div>
    </div>
    <div class="section">
        <div class="title index">03. Contact</div>
            <div class="content">
                <ul>
                    joker1251@naver.com
                </ul>
            </div>
    </div>
</div>