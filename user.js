const {Schema,model} = require('mongoose');
const {createHmac,randomBytes} = require('crypto');

const userSchema = new Schema({
    fullName:{
        type : String,
        required : true
    },
    email : {
        type : String,
        required : true,
        unique : true
    },
    salt: {
        type : String,
        
    },
    profileImageURL : {
        type : String,
        default : "/images/userimage.png"
    },
    role:{
        type : String,
        enum : ["User","Admin"],
        default : "User"
    },
    password : {
        type : String,
        required : true
    
    },},
    {timestamps : true}
)

userSchema.pre('save',function(next){
  const user = this;
  if(!user.isModified('password')){
    return ;
  }

  const salt = randomBytes(16).toString();
  const hashedPassword = createHmac('sha256',salt).update(user.password).digest('hex');

  this.salt = salt;
  this.password = hashedPassword;
  next();
  

});

const User = model('User',userSchema);

module.exports = User;