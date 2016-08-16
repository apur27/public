/*
 * Angular 2 decorators and services
 */
import { Component, OnInit, OnDestroy, Input } from '@angular/core';



/**
 * This component define the breadcrumbs container to hold the breadcrumbs
 * This can be devided further to a sub-component like breadcrumb-item.
 *
 * NOTE: Be aware that angular2 component will create its own wrapper on each
 * sub-component, which might messed up the stylesheet.
 */
@Component({
  selector: 'lb-breadcrumb',
  pipes: [],
  providers: [],
  directives: [],
  templateUrl: './breadcrumb.html'
})
export class BreadcrumbComponent implements OnInit, OnDestroy {
  @Input() public crumbUrl: string;
  @Input() public crumbType: string;
  @Input() public crumbTitle: string;
  @Input() public crumbNumber: string;

  constructor() {
  }

  /**
   * ngOnInit -
   */
  public ngOnInit(): void {
  }

  public ngOnDestroy(): void {
  }
}
