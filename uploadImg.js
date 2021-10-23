
var admin = require("firebase-admin");
console.log(admin.storage, " ", admin.auth);

// CHANGE: The path to your service account
var serviceAccount = require("../python/serviceAccountKey.json");

if (admin.apps.length === 0) {
    admin.initializeApp({
        credential: admin.credential.cert(serviceAccount),
        storageBucket: "optimalpricingsystem.appspot.com"
    });
}

const uploadImg = function(userName,uid,imagename,filename) {   
    console.log("123");
    let bucketName = userName;
    var bucket = admin.storage().bucket();

    // var filename = "GM_x_out.png"

    

    async function uploadFile() {
    const metadata = {
        metadata: {
        // This line is very important. It's to create a download token.
        firebaseStorageDownloadTokens: uid
        },
        contentType: 'image/png',
        cacheControl: 'public, max-age=31536000',
    };
    // console.log(uuid())
    // Uploads a local file to the bucket
    await bucket.upload("public/img/" + imagename, {
        // Support for HTTP requests made with `Accept-Encoding: gzip`
        gzip: true,
        metadata: metadata,
        destination: userName + '/' + filename, 
        // path: userName + '/' + filename,
    });
    
    console.log(`${filename} uploaded.`);

    }

    uploadFile().catch(console.error);


}

// const downloadImg = function(userName) {
//     console.log("hello world")

//     let bucketName = userName;
//     const bucket = admin.storage().bucket();

//     const filename = "GM_x_out.png"

//     //console.log(bucket.file(filename).download);
//     console.log("wa");
//     // app.get('/showsavedresult', async (req, res) => {
//     // // const fileRef = admin.storage().bucket().file('GM_x_out.png');
//     // // const hash = await fileRef.download()
//     // // res.contentType(fileRef.metadata.contentType);
//     // res.end(hash[0], 'binary');
//     //  });

// }




exports.uploadImg = uploadImg;
// exports.downloadImg = downloadImg;