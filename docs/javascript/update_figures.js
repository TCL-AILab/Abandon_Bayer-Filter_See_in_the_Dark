// Figure 2 interactively loads static images, stored locally
function update_figure_2() {
  var period = document.getElementById("Figure_2_period").value;
  var filename = "./images/figure_2/period_" + period + ".svg";
  var image = document.getElementById("Figure_2_image");
  image.src = filename;
}

// Figure 3 interactively loads static images.
// Check if our big images are available locally or remotely:
var big_image_directory = "./../big_images";
var img = new Image();
img.onerror = function() {
  window.big_image_directory = "https://andrewgyork.github.io/publication_template_data/big_images";
  img.onerror = function() {
    window.big_image_directory = "";
    img.onerror = "";
    window.alert("Interactive images not found.");
  }
  img.src = big_image_directory + "/figure_3/period_000004.svg"
}
img.onload = function() {
  console.log("Loading interactive images from: " + big_image_directory)
}
img.src = big_image_directory + "/figure_3/period_000004.svg"

function update_figure_3() {
  var period = document.getElementById("Figure_3_period").value;
  var filename = big_image_directory + "/figure_3/period_" + period + ".svg";
  var image = document.getElementById("Figure_3_image");
  image.src = filename;
}
