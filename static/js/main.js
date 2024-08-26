document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById('search-form');
    const resultsDiv = document.getElementById('results');

    form.addEventListener('submit', (event) => {
        event.preventDefault();
        const query = document.querySelector('input[name="query"]').value;

        fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `query=${encodeURIComponent(query)}`,
        })
        .then(response => response.json())
        .then(data => {
            resultsDiv.innerHTML = ''; // Limpiar resultados anteriores
            if (data.length === 0) {
                resultsDiv.innerHTML = '<p>No se encontraron resultados.</p>';
                return;
            }

            data.forEach(result => {
                const artistElement = document.createElement('div');
                artistElement.classList.add('artist');
                artistElement.innerHTML = `<strong>Artista:</strong> ${result.artist}<br>`;
                result.releases.forEach(release => {
                    artistElement.innerHTML += `<strong>Título:</strong> ${release.title} (${release.date})<br>`;
                });
                resultsDiv.appendChild(artistElement);
            });
        })
        .catch(error => console.error('Error:', error));
    });
});
