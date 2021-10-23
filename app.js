//app.js
const express = require('express')
const app = express()
const port = 3000
const { spawn } = require('child_process')
var dataString = ''
let dataToSend
let largeDataSet = []
let c = []
let selectedConstraintIndex
var user = ""
let name = []
var bfilename = ""
var sfilename = ""
let histData
var isAdded = 0
const uuid = require('uuid-v4');
const prompt = require('prompt-sync')();
const readline = require('readline');



// Firebase App (the core Firebase SDK) is always required and
// must be listed before other Firebase SDKs
var firebase = require('firebase/app')
var admin = require("firebase-admin")
var serviceAccount = require("../python/serviceAccountKey.json")

// Add the Firebase products that you want to use
require('firebase/auth')
require('firebase/firestore')
require('firebase/database');

// admin.initializeApp();

var firebaseConfig = {
  // credential: admin.credential.cert(serviceAccount),
  apiKey: 'AIzaSyDRuY0tzhaLNvTGk4EUx322Lqvn3TuidUg',
  authDomain: 'optimalpricingsystem.firebaseapp.com',
  databaseURL: 'https://optimalpricingsystem-default-rtdb.firebaseio.com',
  projectId: 'optimalpricingsystem',
  storageBucket: 'gs://optimalpricingsystem.appspot.com',
  // messagingSenderId: "65211879809",
  appId: '1:716554695947:web:c67f37c873a749ce9f5e0e',
}

// Initialize Firebase
const firebaseApp = firebase.initializeApp(firebaseConfig)

// for file upload
const multer = require('multer')
const path = require('path')
const helpers = require('./helpers')
const saveImg = require('./uploadImg')

// for constraint
const bodyParser = require('body-parser')
app.use(bodyParser.urlencoded({ extended: true }))

app.use(express.static('./public'))
app.use('/css', express.static(__dirname + './public/css'))
app.use('/js', express.static(__dirname + './public/js'))
app.use('/img', express.static(__dirname + './public/img'))

// Set Views
app.set('views', './views')
app.set('view engine', 'ejs')

// middleware that checks for the current user and passes it to the next middleware using the req object
app.use((req, res, next) => {
  const currentuser = firebase.auth().currentUser
  req.currentUser = currentuser
  next()
})


// middelware to check if user is logged in and protect routes that require log in
// pass this middleware to routes that you would like to protect
const loginCheck = (req, res, next) => {
  const currentuser = req.currentUser
  if (!currentuser) {
    res.render('forbidden',{data: priceComparison(dataString)}, currentuser)
  } else {
    req.currentUser = currentuser
    next()
  }
}

// middelware to prevent loggin in users from accessing certain routes such as signup route
// pass this to routes that logged in users should not be able to access
const notLoginCheck = (req, res, next) => {
  const currentuser = req.currentUser
  if (currentuser) {
    res.render('alreadyloggedin',{data: priceComparison(dataString)}, currentuser)
  } else {
    next()
  }
}


app.get('/', (req, res) => {
  currentUser = req.currentUser
  res.render('home', { data: priceComparison(dataString), bfile: bfilename, sfile: sfilename , constraint: c, currentUser, isAdded })
})

app.get('/about', (req, res) => {
  res.render('about', {data: priceComparison(dataString), currentUser })
})

app.get('/visualisation', (req, res) => {
  currentUser = req.currentUser
  res.render('visualisation', { data: priceComparison(dataString), currentUser })
})

app.get('/constraint', (req, res) => {
  res.render('constraint', { data: priceComparison(dataString), constraint: c, bfilename, sfilename, currentUser })
})

app.post('/shareSuccessful', (req, res) => {

  data = JSON.parse(jsonString)
  var dataNames = Object.keys(data);
  var name = dataNames[selectedText];
  console.log(selectedText)

  var text = data[name].text;
  var id = data[name].id;
  var filename = data[name].filename;
  var imageName = data[name].imageName;
  var bfile = data[name].bfile;
  var sfile = data[name].sfile;
  var timestamp = data[name].timestamp;
  var constraint = data[name].constraint;
  // console.log(id)
  // console.log(bfilename)

  var message = {id, filename, imageName, text, bfile, sfile, timestamp, constraint};
    

  currentUser = req.currentUser
  let userEmailSelected = '';
  let userEmailSelectedArray = new Array();
  userEmailSelected = req.body.emails;
  userEmailSelectedArray = userEmailSelected.split(/[ ,]+/)

  //process the sharing
  //get the data...
  let userNameList = new Array();
  var userList = getUserList();
  var userToBeAdded = new Array();
  for(var i = 0 ; i < userList.length; i++){
    for(var j = 0 ; j < userEmailSelectedArray.length; j++){
      if(userList[i].userEmail == userEmailSelectedArray[j]){
        userToBeAdded.push(userList[i]);

        //share with the user that match the record..
        //store data + image
        console.log(userList[i].userName)
        var messageRef = firebase.database().ref(userList[i].userName).child('Data').push(message);

        console.log(userList[i])
        // console.log(uid)
        console.log(userList[i].imageName)
        console.log(userList[i].filename)
        saveImg.uploadImg(userList[i].userName,id,imageName,filename);        
      }  
    }
  }

  console.log(userToBeAdded)

  res.render('shareFile', { data: priceComparison(dataString), currentUser })
})

// route for see saved data record page
// app.get('/showsavedresult', (req, res) => { 
//   res.render('showsavedresult', { data: priceComparison(dataString), currentUser, data: arrayToReturn, filename: filename })  
// })

// route for log in page
app.get('/login', notLoginCheck, (req, res) => {
  res.render('login', { data: priceComparison(dataString), errorMessage: null, currentUser })
})

// route for signup page
app.get('/signup', notLoginCheck, (req, res) => {
  res.render('signup', { data: priceComparison(dataString), errorMessage: null, currentUser })
})

// logged in user can go to this route to log out.
app.get('/logout', loginCheck, (req, res) => {
  firebase.auth().signOut()
  data = ""
  res.redirect('/')
})

// when signup form is submitted, data is posted to this route to handle the registration of the new user.
app.post('/handlesignup', (req, res) => {
  const useremail = req.body.uemail
  const pw = req.body.psw
  const username = req.body.uname
  const usernric = req.body.unric
  const userDOB = req.body.uDOB
  const userAddress = req.body.uaddress
  const userCompany = req.body.ucompany
  const userPhone = req.body.uphone

  firebase
    .auth()
    .createUserWithEmailAndPassword(useremail, pw)
    .then((userCredential) => {
      // Signed in
      user = userCredential.user
      console.log(user)
      user.updateProfile({
        displayName: username,
        phoneNumber: userPhone,
        company: userCompany
      }).then(function() {
        // Update successful.
      }).catch(function(error) {
        // An error happened.
      });
      res.redirect('/User')
    })
    .catch((error) => {
      var errorCode = error.code
      var errorMessage = error.message
      res.render('signup', { data: priceComparison(dataString), errorMessage, currentUser })
    })

    let userData = {
      userName: username,
      userNric: usernric,
      userDOB: userDOB,
      userAddress: userAddress,
      userPhone: userPhone,
      userCompany: userCompany,
      userEmail: useremail,
    }
    // firebase.database().ref(user.displayName).child('Data').push(message);
    let ref = firebase.database().ref('Users')
    ref.child(username).set(userData)
})

// when login form is submitted, data is posted to this route to handle the log in.
app.post('/handlelogin', function (req, res) {
  const useremail = req.body.uemail
  const pw = req.body.psw

  firebase
    .auth()
    .signInWithEmailAndPassword(useremail, pw)
    .then((userCredential) => {
      // Signed in
      user = userCredential.user
      res.redirect('/User')
    })
    .catch((error) => {
      var errorCode = error.code
      var errorMessage = error.message
      res.render('login', { data: priceComparison(dataString), errorMessage, currentUser })
    })
})

app.post('/handleresetpassword', (req, res) => {
  const email = req.body.email
  firebase
    .auth()
    .sendPasswordResetEmail(email)
    .then(function () {
      res.render('resetpassword', { data: priceComparison(dataString),
        message: 'You have been sent an email to reset your password.', currentUser
      })
    })
    .catch(function (error) {
      res.render('resetpassword', { data: priceComparison(dataString), message: error.message, currentUser })
    })
})

app.get('/resetpassword', (req, res) => {
  res.render('resetpassword', { data: priceComparison(dataString), message: null })
})

app.get('/index', (req, res) => {
  currentUser = req.currentUser
  if (dataString != '') {
    res.render('index', {
      data: priceComparison(dataString),
      constraint: c,
      currentUser,
    })
  } else {
    res.render('home', { data: priceComparison(dataString), bfile: bfilename, sfile: sfilename , constraint: c, currentUser, isAdded })
  }
})

// async function readJSONHist(){
//   await getHist();
//   jsonString = fs.readFileSync(path.resolve(__dirname, 'public/js/data.json'))
//   data = JSON.parse(jsonString)
//   return data
// }

app.get('/history', (req, res) => {
  var data
  try {
    getHist();
    jsonString = fs.readFileSync(path.resolve(__dirname, 'public/js/'+user.displayName+'.json'))
    data = JSON.parse(jsonString)
    var dataNames = Object.keys(data);
    var count = Object.keys(data).length;
    for (var j=0; j<count; j++) {
        var name = dataNames[j];
        var value = data[name];
    }
  } catch(err) {
    console.log(err)
    return
  }
  res.render('history', { data: priceComparison(dataString), currentUser, histData : data, dataNames, count})
})



app.listen(port, () => {
  console.log(`express server running on http://localhost:${port}`)
})

var ref = firebase.database().ref(user.displayName);
var messagesRef = ref.child('Data');

// For file upload
var storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, '../data/')
  },

  // By default, multer removes file extensions so let's add them back
  filename: function (req, file, cb) {
    cb(null, file.originalname)
    name += file.originalname + ' '
  },
})

const upload = multer({ storage: storage, fileFilter: helpers.csvFilter })

    
app.post(
  '/upload-file',
  upload.fields([
    { name: 'baseFile', maxCount: 1 },
    { name: 'salesFile', maxCount: 1 },
  ]),
  function (req, res, next) {

      const baseFile = req.files.baseFile[0]
      const salesFile = req.files.salesFile[0]
      bfilename = baseFile.originalname
      sfilename = salesFile.originalname

      // spawn new child process to call the python script
      // format for constraint = [([c1,c2,c3,....,c20],b)]
      var constraintArray = ""
      var constraintB = ""
      var constraintToSend = ""

      if(c.length != 0){
        for (var i = 0; i < c.length; i++) {
          for(j = 0; j < c[i].length; j++){
            if (j == c[i].length-1) {
              constraintB = c[i][j].toString();
            }else if(j == c[i].length-2){
              constraintArray = constraintArray.concat(c[i][j].toString());
              // constraintArray = constraintArray.concat("]");
            }else{
              constraintArray = constraintArray.concat(c[i][j].toString());
              constraintArray = constraintArray.concat(",");
            }
          }
        }
        console.log(c)
        console.log(constraintB)
  
        constraintToSend = constraintArray+","+constraintB
        console.log(constraintToSend)

      }else{
        constraintToSend = null;
      }

      //Spawn python script with 3 arguments sent over
      const python = spawn('python', [
        'PricingCode_MDM.py',
        bfilename,
        sfilename,
        constraintToSend])

      // collect data from script
      python.stdout.on('data', function (data) {
        console.log('Pipe data from python script ...')
        largeDataSet.push(data)
      })

      // in close event we are sure that stream is from child process is closed
      python.on('close', (code) => {
        console.log(`child process close all stdio with code ${code}`)
        // send data to browser
        dataString = largeDataSet.join()
        res.render('index', { data: priceComparison(dataString), constraint: c, currentUser })
      })
  }
)

// function retrieveData(){

// }

var selectedText;
var jsonString;

app.post('/showsaved', function (req, res) {
  selectedText = parseInt(req.body.selectedText)
  jsonString = fs.readFileSync('public/js/'+user.displayName+'.json')

  data = JSON.parse(jsonString)
  var dataNames = Object.keys(data);
  var name = dataNames[selectedText];
  var value = data[name].text;
  var arrayToReturn = value.split(/[ ,]+/)
  var id = data[name].id;
  var filename = data[name].filename;
  // console.log(currentUser.displayName)
  // console.log(name)
  //get image
  // saveImg.downloadImg("ting");
  
  res.render('showsavedresult', {id: id, filename: filename, uname: currentUser.displayName, data: priceComparison(dataString), currentUser, data: arrayToReturn , date: data[name].timestamp, filename:filename})  
})

app.post('/history', function(req,res){
  res.redirect('/history')
})

app.post('/shareFile', function (req, res) {
  let userList = getUserList();
  var userEmailList = new Array();
  console.log(userList)
  for(var i = 0 ; i < userList.length; i++){
      userEmailList.push(userList[i].userEmail);
  }
  console.log(userEmailList)
  res.render('shareFile',{data: priceComparison(dataString),currentUser,userEmailList})
})

app.post('/submitConstraint', function (req, res) {
  c.push(req.body.constraint)
  bfilename = req.body.bfilename
  sfilename = req.body.sfilename
  isAdded += 1;
  res.render('home', { data: priceComparison(dataString), bfile: bfilename, sfile: sfilename , constraint: c, currentUser, isAdded })
})

app.post('/deleteConstraint', function (req, res) {
  const toDeleteIndex = parseInt(req.body.isConstraint)
  for (var i = 0; i < c.length; i++) {
    if (i == toDeleteIndex) {
      c.splice(toDeleteIndex, 1)
    }
  }
  isAdded -= 1;
  res.render('home', { data: priceComparison(dataString), bfile: bfilename, sfile: sfilename, constraint: c, currentUser, isAdded })
})

app.post('/saveResult', function (req, res) {

  var constraintStored = ""
  var dataStored = ""
  const filename = req.body.output;
  console.log(filename)

  for(i = 0; i < c.length; i++){
    for(j = 0; j < c[i].length; j++){
      constraintStored = constraintStored.concat(c[i][j].toString());
      constraintStored = constraintStored.concat(",");
    }
  }

  for(i = 0; i < priceComparison(dataString).length-1; i++){
    dataStored = dataStored.concat(priceComparison(dataString)[i].toString());
    dataStored = dataStored.concat(",");
  }

  
  if(currentUser != null){
    try{
    //save img record to firebase
    const uid = uuid()
    var imageName = "GM_x_out.png";
    saveImg.uploadImg(user.displayName,uid,imageName,filename);

    var date = new Date().toISOString().replace(/T/, ' ').replace(/\..+/, '')

    //save data record to firebase
    var message = {id: uid, filename:filename, imageName: imageName, text: dataStored, bfile: bfilename, sfile: sfilename , timestamp: date, constraint: constraintStored};
    var messageRef = firebase.database().ref(user.displayName).child('Data').push(message);

    }catch(err) {
      console.log("failed saving file to database")
      return
    }
    
    getHist();
    var data
    try {
      jsonString = fs.readFileSync('public/js/'+user.displayName+'.json')
      data = JSON.parse(jsonString)
      var dataNames = Object.keys(data);
      var count = Object.keys(data).length;
      for (var j=0; j<count; j++) {
          var name = dataNames[j];
          var value = data[name];
      }
    } catch(err) {
      console.log(err)
      return
    }
    
    res.render('history', { data: priceComparison(dataString), currentUser, histData : data, dataNames, count })
  }else{
    res.render('login', {data: priceComparison(dataString), errorMessage: null, currentUser})
  }
  
})


const fs = require('fs');
const { SSL_OP_SSLEAY_080_CLIENT_DH_BUG } = require('constants')

// async function readHist(){
//   let dataHist
//   const snapshot = await firebase.database().ref(user.displayName).child('Data').orderByKey().once('value')
//   console.log("readTest")
//   console.log(snapshot.val())
//   return snapshot.val()
// }

async function getHist(){
  let data
  await firebase.database().ref(user.displayName).child('Data').orderByKey().on('value', function(snap){
      data = JSON.stringify(snap.val(), null, 10);
      fs.writeFileSync('public/js/'+user.displayName+'.json', data, (err) => {
        if (err) throw err;
        console.log('Data written to file');
      });      
    })
}

function getUserList(){
  let count = 0;
  let userList = new Array()
  let ref = firebase.database().ref('Users')
  ref.on('value', function(snap){
    snap.forEach((data) => {
      userList.push(data.val());
      // console.log(data.val())
    });
  })
  // console.log(userList)
  return userList;
}


app.get('/User', loginCheck, (req, res) => {
  currentUser = req.currentUser

  if(priceComparison(dataString)!='undefined'){
    res.render('index', {
      data: priceComparison(dataString), constraint: c, currentUser,
    })
  }else{
    res.render('home', {
      data: priceComparison(dataString), bfile: bfilename, sfile: sfilename, constraint: c, currentUser, isAdded
    })
  }
})

// To format the output to get what we need only
var priceComparison = function (data) {
  let strToReturn = ''
  const regex = /BasePrice.*/g
  pos = data.search(regex)
  for (var i = pos; i < data.length; i++) {
    strToReturn += data[i]
  }

  var arrayToReturn = new Array()
  strToReturn = strToReturn.replace(/\r?\n|\r/g, ',')
  arrayToReturn = strToReturn.split(/[ ,]+/)

  return arrayToReturn
}