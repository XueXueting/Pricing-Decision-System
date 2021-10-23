const callPy = function(fname){

    const { spawn } = require('child_process')
    var dataString;
    let dataToSend
    let largeDataSet = []

    // spawn new child process to call the python script
    const python = spawn('python', ['python/PricingCode_MDM.py', fname, "name"])

    // collect data from script
    python.stdout.on('data', function (data) {
    //console.log(name);
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
    return dataString;

    })
};

exports.callPy = callPy;
