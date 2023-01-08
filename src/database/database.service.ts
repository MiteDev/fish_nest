import { Injectable } from '@nestjs/common';
import * as mysql from 'mysql2/promise';

@Injectable()
export class DatabaseService {
  public CP: mysql.Pool;
  constructor() {
    this.CP = mysql.createPool({
      host: process.env.DB_HOST as string,
      user: process.env.DB_USER as string,
      password: process.env.DB_PASSWORD as string,
      port: parseInt(process.env.DB_PORT as string) ?? 3306,
      database: process.env.DATABASE as string,
    });

    console.log(`Database connected`);
  }
}
