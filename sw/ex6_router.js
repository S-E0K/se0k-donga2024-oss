const http = require('http');

const server = http.createServer((req, res) => {
    const url = req.url;
    // console.log('req:', req)
    // console.log('url:', url)

    if (url === '/') {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end('<h1>Hello, Node.js!</h1>');
    } else if (url === '/about') {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end('<h1>About Page</h1><p>This is a simple routing example.</p>');
    } else {
        res.writeHead(404, { 'Content-Type': 'text/html' });
        res.end('<h1>404 Not Found</h1>');
    }
});

server.listen(3000, () => {
    console.log('Server is running at http://localhost:3000');
});
