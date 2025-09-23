// static/js/scripts.js
function animateAndGo(url){
  const overlay = document.getElementById("loadingOverlay");
  overlay.style.display = "flex";
  // opção: pequeno delay para mostrar animação
  setTimeout(()=> {
    window.location.href = url;
  }, 700);
}
