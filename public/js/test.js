// const express = require('express')
const { spawn } = require('child_process')
// const app = express()
// const port = 3000
var dataString;

// app.use(express.static('python/public'))
// app.use('/css', express.static(__dirname + 'python/public/css'))
// app.use('/js', express.static(__dirname + 'python/public/js'))
// app.use('/img', express.static(__dirname + 'python/public/img'))


// // Set Views
// app.set('views','python/views')
// app.set('view engine', 'ejs')

app.get('/', (req, res) => {

  let dataToSend
  let largeDataSet = []
  // spawn new child process to call the python script
    const python = spawn('python', ['python/PricingCode_MDM.py'])

  // collect data from script
  python.stdout.on('data', function (data) {
    console.log('Pipe data from python script ...')
    //dataToSend =  data;
    //dataString += data.toString() + '<br/>';
    largeDataSet.push(data)
  })

  // in close event we are sure that stream is from child process is closed
  python.on('close', (code) => {
    console.log(`child process close all stdio with code ${code}`)
    // send data to browser
    //var arrayToTable = require('array-to-table')
    //res.send(largeDataSet.join())
    dataString = largeDataSet.join()
    console.log(dataString)
    res.render('index', {data: removeNoisesFromLog(dataString)})
    
  })
})

app.get('/visualisation', (req, res) => {
  res.render('visualisation')
})

app.get('/index', (req, res) => {
  res.render('index', {data: removeNoisesFromLog(dataString)})
})

app.listen(port, () => {
  console.log(`App listening on port ${port}!`)
})

var removeNoisesFromLog = function(data){
  let strToReturn = "";
  const regex = /BasePrice.*/g;
  pos = data.search(regex);
  for(var i = pos ; i < data.length; i++){
    strToReturn += data[i];
  }

  //TESTING FOR TABLE
  var arrayToReturn = new Array();
  strToReturn = strToReturn.replace(/\r?\n|\r/g, ",");
  arrayToReturn = strToReturn.split(/[ ,]+/);

  return arrayToReturn;
}
