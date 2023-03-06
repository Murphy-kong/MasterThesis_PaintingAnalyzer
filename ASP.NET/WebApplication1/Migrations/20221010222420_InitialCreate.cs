using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace WebApplication1.Migrations
{
    public partial class InitialCreate : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "Notifications",
                columns: table => new
                {
                    NotificationID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    Content = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    ReleaseDate = table.Column<DateTime>(type: "datetime2", nullable: false),
                    Type = table.Column<string>(type: "nvarchar(max)", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Notifications", x => x.NotificationID);
                });

            migrationBuilder.CreateTable(
                name: "Users",
                columns: table => new
                {
                    UserID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    Role = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    UserName = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    Email = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    PasswordHash = table.Column<byte[]>(type: "varbinary(max)", nullable: false),
                    PasswordSalt = table.Column<byte[]>(type: "varbinary(max)", nullable: false),
                    ReleaseDate = table.Column<DateTime>(type: "datetime2", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Users", x => x.UserID);
                });

            migrationBuilder.CreateTable(
                name: "Images",
                columns: table => new
                {
                    ImageID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    UserID = table.Column<int>(type: "int", nullable: false),
                    Occupation = table.Column<string>(type: "nvarchar(50)", nullable: false),
                    ImageName = table.Column<string>(type: "nvarchar(100)", nullable: false),
                    ReleaseDate = table.Column<DateTime>(type: "datetime2", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Images", x => x.ImageID);
                    table.ForeignKey(
                        name: "FK_Images_Users_UserID",
                        column: x => x.UserID,
                        principalTable: "Users",
                        principalColumn: "UserID",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "NotificationModelUserModel",
                columns: table => new
                {
                    NotificationsNotificationID = table.Column<int>(type: "int", nullable: false),
                    UsersUserID = table.Column<int>(type: "int", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_NotificationModelUserModel", x => new { x.NotificationsNotificationID, x.UsersUserID });
                    table.ForeignKey(
                        name: "FK_NotificationModelUserModel_Notifications_NotificationsNotificationID",
                        column: x => x.NotificationsNotificationID,
                        principalTable: "Notifications",
                        principalColumn: "NotificationID",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_NotificationModelUserModel_Users_UsersUserID",
                        column: x => x.UsersUserID,
                        principalTable: "Users",
                        principalColumn: "UserID",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "Analyzes",
                columns: table => new
                {
                    AnalyzeID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    ImageID = table.Column<int>(type: "int", nullable: true),
                    ML_Model = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    ReleaseDate = table.Column<DateTime>(type: "datetime2", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Analyzes", x => x.AnalyzeID);
                    table.ForeignKey(
                        name: "FK_Analyzes_Images_ImageID",
                        column: x => x.ImageID,
                        principalTable: "Images",
                        principalColumn: "ImageID",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "Session_Artist_Accuracy",
                columns: table => new
                {
                    ArtistID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    ArtistName = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    Accuracy = table.Column<float>(type: "real", nullable: false),
                    AnalyzeID = table.Column<int>(type: "int", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Session_Artist_Accuracy", x => x.ArtistID);
                    table.ForeignKey(
                        name: "FK_Session_Artist_Accuracy_Analyzes_AnalyzeID",
                        column: x => x.AnalyzeID,
                        principalTable: "Analyzes",
                        principalColumn: "AnalyzeID",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "Session_Genre_Accuracy",
                columns: table => new
                {
                    GenreID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    GenreName = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    Accuracy = table.Column<float>(type: "real", nullable: false),
                    AnalyzeID = table.Column<int>(type: "int", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Session_Genre_Accuracy", x => x.GenreID);
                    table.ForeignKey(
                        name: "FK_Session_Genre_Accuracy_Analyzes_AnalyzeID",
                        column: x => x.AnalyzeID,
                        principalTable: "Analyzes",
                        principalColumn: "AnalyzeID",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "Session_Style_Accuracy",
                columns: table => new
                {
                    StyleID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    StyleName = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    Accuracy = table.Column<float>(type: "real", nullable: false),
                    AnalyzeID = table.Column<int>(type: "int", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Session_Style_Accuracy", x => x.StyleID);
                    table.ForeignKey(
                        name: "FK_Session_Style_Accuracy_Analyzes_AnalyzeID",
                        column: x => x.AnalyzeID,
                        principalTable: "Analyzes",
                        principalColumn: "AnalyzeID",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_Analyzes_ImageID",
                table: "Analyzes",
                column: "ImageID");

            migrationBuilder.CreateIndex(
                name: "IX_Images_UserID",
                table: "Images",
                column: "UserID");

            migrationBuilder.CreateIndex(
                name: "IX_NotificationModelUserModel_UsersUserID",
                table: "NotificationModelUserModel",
                column: "UsersUserID");

            migrationBuilder.CreateIndex(
                name: "IX_Session_Artist_Accuracy_AnalyzeID",
                table: "Session_Artist_Accuracy",
                column: "AnalyzeID");

            migrationBuilder.CreateIndex(
                name: "IX_Session_Genre_Accuracy_AnalyzeID",
                table: "Session_Genre_Accuracy",
                column: "AnalyzeID");

            migrationBuilder.CreateIndex(
                name: "IX_Session_Style_Accuracy_AnalyzeID",
                table: "Session_Style_Accuracy",
                column: "AnalyzeID");
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "NotificationModelUserModel");

            migrationBuilder.DropTable(
                name: "Session_Artist_Accuracy");

            migrationBuilder.DropTable(
                name: "Session_Genre_Accuracy");

            migrationBuilder.DropTable(
                name: "Session_Style_Accuracy");

            migrationBuilder.DropTable(
                name: "Notifications");

            migrationBuilder.DropTable(
                name: "Analyzes");

            migrationBuilder.DropTable(
                name: "Images");

            migrationBuilder.DropTable(
                name: "Users");
        }
    }
}
