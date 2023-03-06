using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Authorization;
using WebApplication1.Models;
using System.Security.Claims;


namespace WebApplication1.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class NotificationController : ControllerBase
    {
        private readonly PaintingAnalyzerDbContext _context;

        public NotificationController(PaintingAnalyzerDbContext context)
        {
            _context = context;
        }

        // GET: api/Notification
        [HttpGet]
        public async Task<ActionResult<IEnumerable<NotificationModel>>> GetNotifications()
        {
            /*  var other = new NotificationHelpClass(_context);

              var testjson =  new NotificationHelpClass.AppJson()
              {
                  Sender = "Eddy",
                  Empfaenger = new[] { "Daniela", "Melissa", "Markus" },
                  content =  "Das ist eine schoene Nachricht 3",
                  releasedate =  "2022-06-21T18:10:00"
              };

              Console.WriteLine("Bis hierhin hat geklappt");

              await other.PostUserNotificationModel(testjson);*/

            if (_context.Notifications == null)
          {
              return NotFound();
          }
            return await _context.Notifications.ToListAsync();
        }

        // GET: api/Notification/5
       [HttpGet("{id}")]
        public async Task<ActionResult<NotificationModel>> GetNotificationModel(int id)
        {
   
            if (_context.Notifications == null)
          {
              return NotFound();
          }
            var notificationModel = await _context.Notifications.FindAsync(id);

            if (notificationModel == null)
            {
                return NotFound();
            }

            return notificationModel;
        }

        
        // PUT: api/Notification/5
        // To protect from overposting attacks, see https://go.microsoft.com/fwlink/?linkid=2123754
        [HttpPut("{id}")]
        public async Task<IActionResult> PutNotificationModel(int id, NotificationModel notificationModel)
        {
            if (id != notificationModel.NotificationID)
            {
                return BadRequest();
            }

            _context.Entry(notificationModel).State = EntityState.Modified;

            try
            {
                await _context.SaveChangesAsync();
            }
            catch (DbUpdateConcurrencyException)
            {
                if (!NotificationModelExists(id))
                {
                    return NotFound();
                }
                else
                {
                    throw;
                }
            }

            return NoContent();
        }



        // POST: api/Notification
        // To protect from overposting attacks, see https://go.microsoft.com/fwlink/?linkid=2123754
        [HttpPost]
        public async Task<ActionResult<NotificationModel>> PostNotificationModel(NotificationModel notificationModel)
        {
    
          if (_context.Notifications == null)
          {
              return Problem("Entity set 'PaintingAnalyzerDbContext.Notifications'  is null.");
          }
            _context.Notifications.Add(notificationModel);
            await _context.SaveChangesAsync();

            return CreatedAtAction("GetNotificationModel", new { id = notificationModel.NotificationID }, notificationModel);
        }

        
        // DELETE: api/Notification/5
        [HttpDelete("{id}")]
        public async Task<IActionResult> DeleteNotificationModel(int id)
        {
            if (_context.Notifications == null)
            {
                return NotFound();
            }
            var notificationModel = await _context.Notifications.FindAsync(id);
            if (notificationModel == null)
            {
                return NotFound();
            }

            _context.Notifications.Remove(notificationModel);
            await _context.SaveChangesAsync();

            return NoContent();
        }

        [HttpDelete("UsersNotificationDelete"), Authorize(Roles = "Admin,Registered User")]
        public async Task<IActionResult> UsersNotificationDelete(int id)
        {
            var user = await _context.Users
               .Where(c => c.UserName == User.FindFirstValue(ClaimTypes.Name))
               .Include(i => i.Notifications)
               .FirstOrDefaultAsync();


            if (EqualityComparer<UserModel>.Default.Equals(user, default(UserModel)))
                return BadRequest(" Username not found.");

            var notification = user.Notifications
                .Where(c => c.NotificationID == id)
                .FirstOrDefault();

            if (EqualityComparer<NotificationModel>.Default.Equals(notification, default(NotificationModel)))
                return BadRequest(" Notification not found.");

            if (_context.Notifications == null)
            {
                return NotFound();
            }
            var notificationModel = await _context.Notifications.FindAsync(id);
            if (notificationModel == null)
            {
                return NotFound();
            }

            _context.Notifications.Remove(notificationModel);
            await _context.SaveChangesAsync();

            return NoContent();
        }

        private bool NotificationModelExists(int id)
        {
            return (_context.Notifications?.Any(e => e.NotificationID == id)).GetValueOrDefault();
        }
    }
}
/* 
 * List<string> lst = ... // your list containging xx@yy

List<string> _featureNames = new List<string>();

List<string> _projectNames = new List<string>();

lst.ForEach(x => 
{
    string[] str = x.Split('@');
    _featureNames.Add(str[0]);
    _projectNames.Add(str[1]);
}

string[] featureNames = _featureNames.ToArray();

string[] projectNames = _projectNames.ToArray();
 * */