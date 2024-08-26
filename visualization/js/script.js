const output = document.getElementById('output');
const messages = [
    "Loading your music library...",
    "Spotify limiting how many people can use this app at once, so you'll need to wait or try again later.",
    "Analyzing your listening history...",
    "lol",
    "omg",
    "okay hold up"
];

function typeWriter(text, i = 0) {
    if (i < text.length) {
        output.innerHTML += text.charAt(i);
        i++;
        setTimeout(() => typeWriter(text, i), 50);
    } else {
        output.innerHTML += '<br><br>';
        displayNextMessage();
    }
}

function displayNextMessage() {
    if (messages.length > 0) {
        const message = messages.shift();
        typeWriter(message);
    }
}

displayNextMessage();

function addGraph() {
    const ctx = document.getElementById('myChart');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
            datasets: [{
                label: '# of Votes',
                data: [12, 19, 3, 5, 2, 3],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}