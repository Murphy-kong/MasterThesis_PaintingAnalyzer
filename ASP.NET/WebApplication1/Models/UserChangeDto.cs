namespace WebApplication1.Models
{
    public class UserChangeDto
    {
        public string Username { get; set; } = string.Empty;

        public string OldPassword { get; set; } = string.Empty;
       
        public string NewPassword { get; set; } = string.Empty;

        public string Email { get; set; } = string.Empty;
    }
}
