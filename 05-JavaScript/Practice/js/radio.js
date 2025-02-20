const form = document.getElementById("shippingForm")
form.addEventListener("submit", function(event) {
    event.preventDefault();

    const selectedShipping = document.querySelector("input[name='shipping']:checked").value;

    const message = "You have selected " + selectedShipping + " shipping";
    document.getElementById("message").textContent = message;
});