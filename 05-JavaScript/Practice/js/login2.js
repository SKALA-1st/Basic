// Get the form element
const message = () => {
    // Get the form element
    const form = document.querySelector('form');
    
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
};