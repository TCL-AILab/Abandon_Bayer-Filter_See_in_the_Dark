`update_figures.js` is specific code, hand-written to animate the figures in this particular publication. If you copy this publication as a template, you'll have to rewrite this file yourself (or just use static figures).

You should also delete or modify the `google-analytics/analytics.js` script, since the tracking code is specific to me.

`reference_list/reference_list.js` is more general code; it's intended to search the document for citations formatted in a particular way, and append a (not too ugly) list of references to the end of the document. It's intended to stay the same every time the template is used, unless the user wants to do their reference list a different way.

`scale-fix/scale-fix.js` came with the template that I based my work on, ['dinky' by 'broccolini'](https://github.com/broccolini/dinky). I believe it helps the document scale correctly for different size screens. It's possible that it's not doing anything, and it's just cruft, but I haven't taken the time to understand. Incidentally, 'broccolini' mentioned in their original work that 'attribution is not necessary but it is appreciated', so I put an attribution in a comment in index.html; if you copy this publication as a template, I'd appreciate it if this attribution remains.

`html5shiv/html5shiv.js` is from [HTML5 boilerplate](https://html5boilerplate.com/). As I understand it, this file helps make HTML5 work with old versions of IE, but I haven't tested this extensively.

`python-highlighting/prism.js` is from [prismjs.com/](http://prismjs.com/); the version I included allows python sytnax highlighting for code inside `<pre><code>` tags. If someone wanted syntax highlighting for another language, they should replace `prism.js` with a fresh copy downloaded from `prismjs.com`

`Minimal-MathJax/` is to allow math typesetting, which I imagine many template users would want.
