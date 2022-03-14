document.addEventListener("DOMContentLoaded", function(event) {
  already_included = "";
  list_of_refs = document.createElement("ol");
  var links = document.getElementsByClassName("citation");
  for(var i = 0; i < links.length; i++){
    if (!(already_included.indexOf(links[i].innerText) >= 0)) {
      already_included = already_included.concat(links[i].innerText);
      li = document.createElement("li");
      li.appendChild(document.createTextNode("["));
      lnk = document.createElement("a");
      lnk.href = links[i].href;
      lnk.innerText = links[i].innerText;
      li.appendChild(lnk);
      li.appendChild(document.createTextNode(
        "] "+links[i].title+" "+links[i].href));
      list_of_refs.appendChild(li);
    }
  }
  if (links.length > 0) {
    sections = document.body.getElementsByTagName("section");
    last_section = sections[sections.length - 1];
    heading = document.createElement("h3");
    heading.appendChild(document.createTextNode("References"));
    last_section.appendChild(heading);
    last_section.appendChild(list_of_refs);
  }
});
