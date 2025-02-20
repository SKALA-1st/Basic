// Get the form element
const form = document.getElementById("loginForm");

// Add an event listener to the form
form.addEventListener("submit", function(event) {
    // Prevent the form from submitting
    event.preventDefault();

    // Get the username and password values
    const username = form.username.value;
    const password = form.password.value;

    // Check if the username and password are empty
    if (!username || !password) {
        alert("Please enter a username and password");
        return;
    }

    // Display the username and password on the document
    const loginDetails = `
        <div>
            <h2>Login Details</h2>
            <p>Username: ${username}</p>
            <p>Password: ${password}</p>
        </div>
    `;

    document.body.innerHTML += loginDetails;
});