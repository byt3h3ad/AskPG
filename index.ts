const links = document.querySelectorAll("a");
const set = new Set();
links.forEach((link) => {
  const href = link.getAttribute("href");
  if (href) {
    set.add(href);
  }
});
const uniqueLinks = Array.from(set);
console.log(uniqueLinks);
